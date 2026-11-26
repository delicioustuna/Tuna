using Tuna.Integration.Internal;
using Tuna.LinearAlgebra;

namespace Tuna.Integration;

/// <summary>
/// Provides one-dimensional adaptive numerical integration and trapezoidal rules for sampled data.
/// <para>1変数の適応型数値積分および標本値に対する台形則を提供します。</para>
/// </summary>
/// <remarks>
/// <para>Adaptive quadrature never evaluates the integrand at interval endpoints. Provide interior singularities, discontinuities, or narrow peaks as breakpoints. For infinite intervals whose main contribution lies far from the origin, place a breakpoint near that region.</para>
/// <para>The integrand is evaluated on the calling thread. If an evaluation returns <see cref="double.NaN"/> or infinity, integration terminates with <see cref="IntegrationTermination.NonFiniteValue"/>.</para>
/// <para>適応型求積は区間の端点を評価しません。区間内の特異点、不連続点、狭いピークの位置が分かっている場合は区切り点として指定してください。無限区間で主な寄与が原点から離れている場合も、その近傍に区切り点を置くと精度が向上します。</para>
/// <para>被積分関数は呼び出し元スレッドで評価され、<see cref="double.NaN"/> または無限大が返ると <see cref="IntegrationTermination.NonFiniteValue"/> で終了します。</para>
/// </remarks>
public static class Quadrature
{
    private const int RuleEvaluations = GaussKronrod.Points;

    // これより狭い区間は分割しない（節点が端点と区別できなくなるか、非正規化数領域に近づくため）
    private const double RelativeMinimumWidth = 1024 * GaussKronrod.MachineEpsilon;
    private const double AbsoluteMinimumWidth = 1e-300;

    /// <summary>
    /// Computes the definite integral <c>∫ₐᵇ f(x) dx</c>.
    /// <para>定積分 <c>∫ₐᵇ f(x) dx</c> を計算します。</para>
    /// </summary>
    /// <param name="integrand">The integrand function. / 被積分関数。</param>
    /// <param name="a">The lower integration bound (may be ±∞). / 積分区間の下端（±∞ を指定可能）。</param>
    /// <param name="b">The upper integration bound (may be ±∞). / 積分区間の上端（±∞ を指定可能）。</param>
    /// <param name="options">Integration options, or <see langword="null"/> to use default settings. / 積分の設定。<see langword="null"/> の場合は既定値を使用します。</param>
    /// <returns>The integral value, error estimate, evaluation count, and termination reason. Returns 0 without evaluating the integrand when <paramref name="a"/> equals <paramref name="b"/>, and reverses the sign when <paramref name="a"/> is greater than <paramref name="b"/>. / 積分値、誤差見積もり、評価回数、および終了理由。<paramref name="a"/> と <paramref name="b"/> が等しい場合は被積分関数を評価せずに 0 を返し、<paramref name="a"/> が <paramref name="b"/> より大きい場合は符号を反転します。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="integrand"/> is <see langword="null"/>. / <paramref name="integrand"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="a"/> or <paramref name="b"/> is <see cref="double.NaN"/>. / <paramref name="a"/> または <paramref name="b"/> が <see cref="double.NaN"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">A tolerance is negative or non-finite, or the evaluation limit is smaller than the initial interval count times 15. / 許容誤差が負または有限値でない場合、あるいは評価回数の上限が初期区間数 × 15 未満の場合。</exception>
    public static IntegrationResult Integrate(Func<double, double> integrand, double a, double b, IntegrationOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(integrand);
        return Integrate(new DelegateIntegrand(integrand), a, b, [], options);
    }

    /// <summary>
    /// Computes the definite integral <c>∫ₐᵇ f(x) dx</c> with interior breakpoints.
    /// <para>区切り点を指定して定積分 <c>∫ₐᵇ f(x) dx</c> を計算します。</para>
    /// </summary>
    /// <param name="integrand">The integrand function. / 被積分関数。</param>
    /// <param name="a">The lower integration bound (may be ±∞). / 積分区間の下端（±∞ を指定可能）。</param>
    /// <param name="b">The upper integration bound (may be ±∞). / 積分区間の上端（±∞ を指定可能）。</param>
    /// <param name="breakpoints">Finite interior points of singularity, discontinuity, or slope change in any order. The integrand is not evaluated at these points; duplicates and endpoint values are ignored. / 区間内の特異点、不連続点、または折れ目となる有限の点（順不同）。これらの点では評価されず、重複や端点と一致する値は無視されます。</param>
    /// <param name="options">Integration options, or <see langword="null"/> to use default settings. / 積分の設定。<see langword="null"/> の場合は既定値を使用します。</param>
    /// <returns>The integral value, error estimate, evaluation count, and termination reason. Returns 0 without evaluating the integrand when <paramref name="a"/> equals <paramref name="b"/>, and reverses the sign when <paramref name="a"/> is greater than <paramref name="b"/>. / 積分値、誤差見積もり、評価回数、および終了理由。<paramref name="a"/> と <paramref name="b"/> が等しい場合は被積分関数を評価せずに 0 を返し、<paramref name="a"/> が <paramref name="b"/> より大きい場合は符号を反転します。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="integrand"/> is <see langword="null"/>. / <paramref name="integrand"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="a"/> or <paramref name="b"/> is <see cref="double.NaN"/>, or <paramref name="breakpoints"/> contains a non-finite value. / <paramref name="a"/> または <paramref name="b"/> が <see cref="double.NaN"/> の場合、あるいは <paramref name="breakpoints"/> に有限値でない要素が含まれる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">A breakpoint lies outside the integration interval, a tolerance is negative or non-finite, or the evaluation limit is smaller than the initial interval count times 15. / 区切り点が積分区間外にある場合、許容誤差が負または有限値でない場合、あるいは評価回数の上限が初期区間数 × 15 未満の場合。</exception>
    public static IntegrationResult Integrate(Func<double, double> integrand, double a, double b, ReadOnlySpan<double> breakpoints,
        IntegrationOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(integrand);
        return Integrate(new DelegateIntegrand(integrand), a, b, breakpoints, options);
    }

    /// <summary>
    /// Computes the definite integral <c>∫ₐᵇ f(x) dx</c>.
    /// <para>定積分 <c>∫ₐᵇ f(x) dx</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="TIntegrand">The type of the integrand. / 被積分関数の型。</typeparam>
    /// <inheritdoc cref="Integrate(Func{double, double}, double, double, IntegrationOptions?)"/>
    public static IntegrationResult Integrate<TIntegrand>(TIntegrand integrand, double a, double b, IntegrationOptions? options = null)
        where TIntegrand : IIntegrand =>
        Integrate(integrand, a, b, [], options);

    /// <summary>
    /// Computes the definite integral <c>∫ₐᵇ f(x) dx</c> with interior breakpoints.
    /// <para>区切り点を指定して定積分 <c>∫ₐᵇ f(x) dx</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="TIntegrand">The type of the integrand. / 被積分関数の型。</typeparam>
    /// <inheritdoc cref="Integrate(Func{double, double}, double, double, ReadOnlySpan{double}, IntegrationOptions?)"/>
    public static IntegrationResult Integrate<TIntegrand>(TIntegrand integrand, double a, double b, ReadOnlySpan<double> breakpoints,
        IntegrationOptions? options = null)
        where TIntegrand : IIntegrand
    {
        ArgumentNullException.ThrowIfNull(integrand);
        options ??= IntegrationOptions.Default;
        options.Validate();
        if (double.IsNaN(a))
        {
            throw new ArgumentException("The lower limit must not be NaN.", nameof(a));
        }

        if (double.IsNaN(b))
        {
            throw new ArgumentException("The upper limit must not be NaN.", nameof(b));
        }

        double sign = 1;
        if (a > b)
        {
            (a, b) = (b, a);
            sign = -1;
        }

        foreach (double point in breakpoints)
        {
            if (!double.IsFinite(point))
            {
                throw new ArgumentException("The breakpoints must be finite.", nameof(breakpoints));
            }

            if (point < a || point > b)
            {
                throw new ArgumentOutOfRangeException(nameof(breakpoints), "The breakpoints must lie in the interval.");
            }
        }

        if (a == b)
        {
            return new IntegrationResult(0, 0, 0, IntegrationTermination.Tolerance);
        }

        var pieces = Partition(a, b, breakpoints, out double lowerOrigin, out double upperOrigin);
        options.ValidateBudget(RuleEvaluations * (long)pieces.Count);

        var run = new Adaptive<TIntegrand>(integrand, lowerOrigin, upperOrigin);
        var result = run.Integrate(pieces, options);
        return result with { Value = sign * result.Value };
    }

    /// <summary>
    /// Integrates uniformly spaced samples using the composite trapezoidal rule <c>h * (y₀/2 + y₁ + ... + y_(n-2) + y_(n-1)/2)</c>.
    /// <para>等間隔の標本値に対して合成台形則 <c>h * (y₀/2 + y₁ + ... + y_(n-2) + y_(n-1)/2)</c> で積分を計算します。</para>
    /// </summary>
    /// <param name="y">The sample values. / 標本値。</param>
    /// <param name="spacing">The finite spacing <c>h</c> between adjacent samples; negative values reverse the sign. / 標本間隔 <c>h</c>（有限値。負の場合は符号が反転します）。</param>
    /// <returns>The integral value, or 0 if fewer than two samples are given. Returns <see cref="double.NaN"/> if <paramref name="y"/> contains <see cref="double.NaN"/>, contains both +∞ and −∞, or sums to infinity while <paramref name="spacing"/> is 0. / 積分値。標本数が 2 未満の場合は 0。<see cref="double.NaN"/> を含む場合、+∞ と −∞ の両方を含む場合、または和が無限大かつ <paramref name="spacing"/> が 0 の場合は <see cref="double.NaN"/>。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="spacing"/> is not finite. / <paramref name="spacing"/> が有限値でない場合。</exception>
    public static double Trapezoid(ReadOnlyVectorSpan<double> y, double spacing = 1)
    {
        if (!double.IsFinite(spacing))
        {
            throw new ArgumentOutOfRangeException(nameof(spacing), "The spacing must be finite.");
        }

        int n = y.Length;
        if (n < 2)
        {
            return 0;
        }

        var sum = new CompensatedSum();
        sum.Add(0.5 * y[0]);
        for (int i = 1; i < n - 1; i++)
        {
            sum.Add(y[i]);
        }

        sum.Add(0.5 * y[n - 1]);
        return spacing * sum.Value;
    }

    /// <summary>
    /// Integrates samples at arbitrary coordinates using the composite trapezoidal rule <c>Σ (x_(i+1) - x_i)(y_i + y_(i+1))/2</c>.
    /// <para>指定した座標の標本値に対して合成台形則 <c>Σ (x_(i+1) - x_i)(y_i + y_(i+1))/2</c> で積分を計算します。</para>
    /// </summary>
    /// <param name="y">The sample values. / 標本値。</param>
    /// <param name="x">The sample coordinates, with the same length as <paramref name="y"/>. Need not be sorted; decreasing steps contribute negatively. / 標本座標（<paramref name="y"/> と同じ長さ）。昇順である必要はなく、減少区間は負の寄与になります。</param>
    /// <returns>The integral value, or 0 if fewer than two samples are given. Returns <see cref="double.NaN"/> if any coordinate or sample is <see cref="double.NaN"/>; infinities follow IEEE 754 arithmetic. / 積分値。標本数が 2 未満の場合は 0。座標または標本に <see cref="double.NaN"/> を含む場合は <see cref="double.NaN"/>、無限大の演算は IEEE 754 に従います。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> and <paramref name="y"/> have different lengths. / <paramref name="x"/> と <paramref name="y"/> の長さが異なる場合。</exception>
    public static double Trapezoid(ReadOnlyVectorSpan<double> y, ReadOnlyVectorSpan<double> x)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException("The points and the samples must have the same length.", nameof(x));
        }

        var sum = new CompensatedSum();
        for (int i = 0; i + 1 < y.Length; i++)
        {
            sum.Add((x[i + 1] - x[i]) * (0.5 * y[i] + 0.5 * y[i + 1]));
        }

        return sum.Value;
    }

    /// <summary>
    /// Computes the cumulative trapezoidal integral of uniformly spaced samples into <paramref name="destination"/>.
    /// <para>等間隔の標本値に対する台形則の累積積分を計算し、<paramref name="destination"/> に格納します。</para>
    /// </summary>
    /// <param name="y">The sample values. / 標本値。</param>
    /// <param name="spacing">The finite spacing between adjacent samples. / 標本間隔（有限値）。</param>
    /// <param name="destination">The output vector span, with the same length as <paramref name="y"/>. May overlap <paramref name="y"/>. / 出力先のベクトルスパン（<paramref name="y"/> と同じ長さ）。<paramref name="y"/> と重複していても構いません。</param>
    /// <returns><paramref name="destination"/>, containing the cumulative integral from the first sample (the first element is 0 when non-empty). / 先頭の標本から各標本までの累積積分値を格納した <paramref name="destination"/>（空でない場合、先頭要素は 0）。</returns>
    /// <remarks>
    /// <para><see cref="double.NaN"/>, infinities, and sum overflow follow the same rules as <see cref="Trapezoid(ReadOnlyVectorSpan{double}, double)"/> and propagate to subsequent elements.</para>
    /// <para><see cref="double.NaN"/> や無限大、和のオーバーフローの扱いは <see cref="Trapezoid(ReadOnlyVectorSpan{double}, double)"/> と同じで、以降の要素へ伝播します。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="spacing"/> is not finite. / <paramref name="spacing"/> が有限値でない場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="destination"/> does not have the same length as <paramref name="y"/>. / <paramref name="destination"/> の長さが <paramref name="y"/> と異なる場合。</exception>
    public static VectorSpan<double> CumulativeTrapezoid(ReadOnlyVectorSpan<double> y, double spacing, VectorSpan<double> destination)
    {
        if (!double.IsFinite(spacing))
        {
            throw new ArgumentOutOfRangeException(nameof(spacing), "The spacing must be finite.");
        }

        if (destination.Length != y.Length)
        {
            throw new ArgumentException("The destination must have the same length as the samples.", nameof(destination));
        }

        double[]? copy = Overlaps(y, destination) ? y.ToArray() : null;
        ReadOnlyVectorSpan<double> samples = copy is null ? y : copy;

        var sum = new CompensatedSum();
        for (int i = 0; i < samples.Length; i++)
        {
            if (i > 0)
            {
                sum.Add(0.5 * samples[i - 1] + 0.5 * samples[i]);
            }

            destination[i] = spacing * sum.Value;
        }

        return destination;
    }

    /// <summary>
    /// Computes the cumulative trapezoidal integral of samples at arbitrary coordinates into <paramref name="destination"/>.
    /// <para>指定した座標の標本値に対する台形則の累積積分を計算し、<paramref name="destination"/> に格納します。</para>
    /// </summary>
    /// <param name="y">The sample values. / 標本値。</param>
    /// <param name="x">The sample coordinates, with the same length as <paramref name="y"/>. / 標本座標（<paramref name="y"/> と同じ長さ）。</param>
    /// <param name="destination">The output vector span, with the same length as <paramref name="y"/>. May overlap <paramref name="x"/> or <paramref name="y"/>. / 出力先のベクトルスパン（<paramref name="y"/> と同じ長さ）。<paramref name="x"/> や <paramref name="y"/> と重複していても構いません。</param>
    /// <returns><paramref name="destination"/>, containing the cumulative integral from the first sample (the first element is 0 when non-empty). / 先頭の標本から各標本までの累積積分値を格納した <paramref name="destination"/>（空でない場合、先頭要素は 0）。</returns>
    /// <remarks>
    /// <para><see cref="double.NaN"/>, infinities, and sum overflow follow the same rules as <see cref="Trapezoid(ReadOnlyVectorSpan{double}, ReadOnlyVectorSpan{double})"/> and propagate to subsequent elements.</para>
    /// <para><see cref="double.NaN"/> や無限大、和のオーバーフローの扱いは <see cref="Trapezoid(ReadOnlyVectorSpan{double}, ReadOnlyVectorSpan{double})"/> と同じで、以降の要素へ伝播します。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="x"/>, <paramref name="y"/>, and <paramref name="destination"/> do not all have the same length. / <paramref name="x"/>、<paramref name="y"/>、<paramref name="destination"/> の長さが一致しない場合。</exception>
    public static VectorSpan<double> CumulativeTrapezoid(ReadOnlyVectorSpan<double> y, ReadOnlyVectorSpan<double> x, VectorSpan<double> destination)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException("The points and the samples must have the same length.", nameof(x));
        }

        if (destination.Length != y.Length)
        {
            throw new ArgumentException("The destination must have the same length as the samples.", nameof(destination));
        }

        double[]? copyY = Overlaps(y, destination) ? y.ToArray() : null;
        double[]? copyX = Overlaps(x, destination) ? x.ToArray() : null;
        ReadOnlyVectorSpan<double> samples = copyY is null ? y : copyY;
        ReadOnlyVectorSpan<double> points = copyX is null ? x : copyX;

        var sum = new CompensatedSum();
        for (int i = 0; i < samples.Length; i++)
        {
            if (i > 0)
            {
                sum.Add((points[i] - points[i - 1]) * (0.5 * samples[i - 1] + 0.5 * samples[i]));
            }

            destination[i] = sum.Value;
        }

        return destination;
    }

    /// <summary>
    /// Computes the cumulative trapezoidal integral of uniformly spaced samples and returns a new vector.
    /// <para>等間隔の標本値に対する台形則の累積積分を計算し、新しいベクトルとして返します。</para>
    /// </summary>
    /// <param name="y">The sample values. / 標本値。</param>
    /// <param name="spacing">The finite spacing between adjacent samples. The default is 1. / 標本間隔（有限値）。既定値は 1 です。</param>
    /// <returns>A new vector containing the cumulative integral from the first sample (the first element is 0 when non-empty). / 先頭の標本から各標本までの累積積分値を格納した新しいベクトル（空でない場合、先頭要素は 0）。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="spacing"/> is not finite. / <paramref name="spacing"/> が有限値でない場合。</exception>
    /// <seealso cref="CumulativeTrapezoid(ReadOnlyVectorSpan{double}, double, VectorSpan{double})"/>
    public static Vector<double> CumulativeTrapezoid(ReadOnlyVectorSpan<double> y, double spacing = 1)
    {
        var destination = new Vector<double>(y.Length);
        CumulativeTrapezoid(y, spacing, destination.AsVectorSpan());
        return destination;
    }

    /// <summary>
    /// Computes the cumulative trapezoidal integral of samples at arbitrary coordinates and returns a new vector.
    /// <para>指定した座標の標本値に対する台形則の累積積分を計算し、新しいベクトルとして返します。</para>
    /// </summary>
    /// <param name="y">The sample values. / 標本値。</param>
    /// <param name="x">The sample coordinates, with the same length as <paramref name="y"/>. / 標本座標（<paramref name="y"/> と同じ長さ）。</param>
    /// <returns>A new vector containing the cumulative integral from the first sample (the first element is 0 when non-empty). / 先頭の標本から各標本までの累積積分値を格納した新しいベクトル（空でない場合、先頭要素は 0）。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> and <paramref name="y"/> have different lengths. / <paramref name="x"/> と <paramref name="y"/> の長さが異なる場合。</exception>
    /// <seealso cref="CumulativeTrapezoid(ReadOnlyVectorSpan{double}, ReadOnlyVectorSpan{double}, VectorSpan{double})"/>
    public static Vector<double> CumulativeTrapezoid(ReadOnlyVectorSpan<double> y, ReadOnlyVectorSpan<double> x)
    {
        var destination = new Vector<double>(y.Length);
        CumulativeTrapezoid(y, x, destination.AsVectorSpan());
        return destination;
    }

    private static bool Overlaps(ReadOnlyVectorSpan<double> source, VectorSpan<double> destination) =>
        !source.IsEmpty && source.Data.Overlaps(destination.AsReadOnly().Data);

    private static List<Piece> Partition(double a, double b, ReadOnlySpan<double> breakpoints, out double lowerOrigin, out double upperOrigin)
    {
        bool lowerInfinite = double.IsNegativeInfinity(a);
        bool upperInfinite = double.IsPositiveInfinity(b);

        // 有限の端点と内側の区切り点を昇順・重複なしで収集する
        var knots = new List<double>(breakpoints.Length + 2);
        if (!lowerInfinite)
        {
            knots.Add(a);
        }

        foreach (double point in breakpoints)
        {
            if (point > a && point < b)
            {
                knots.Add(point);
            }
        }

        if (!upperInfinite)
        {
            knots.Add(b);
        }

        knots.Sort();
        for (int i = knots.Count - 1; i > 0; i--)
        {
            if (knots[i] == knots[i - 1])
            {
                knots.RemoveAt(i);
            }
        }

        // (−∞, ∞) で区切り点がなければ 0 で分割する
        if (knots.Count == 0)
        {
            knots.Add(0);
        }

        var pieces = new List<Piece>(knots.Count + 3);
        lowerOrigin = upperOrigin = 0;
        if (lowerInfinite)
        {
            double first = knots[0];
            lowerOrigin = first - 1;
            pieces.Add(new Piece(0, 1, PieceKind.LowerTail));
            if (lowerOrigin < first)
            {
                pieces.Add(new Piece(lowerOrigin, first, PieceKind.Finite));
            }
            else
            {
                lowerOrigin = first;
            }
        }

        for (int i = 0; i + 1 < knots.Count; i++)
        {
            pieces.Add(new Piece(knots[i], knots[i + 1], PieceKind.Finite));
        }

        if (upperInfinite)
        {
            double last = knots[^1];
            upperOrigin = last + 1;
            if (upperOrigin > last)
            {
                pieces.Add(new Piece(last, upperOrigin, PieceKind.Finite));
            }
            else
            {
                upperOrigin = last;
            }

            pieces.Add(new Piece(0, 1, PieceKind.UpperTail));
        }

        return pieces;
    }

    private enum PieceKind : byte
    {
        Finite,
        LowerTail,
        UpperTail,
    }

    private readonly record struct Piece(double T0, double T1, PieceKind Kind);

    private readonly record struct Interval(double T0, double T1, PieceKind Kind, double Value, double Error);

    // ∫_s^∞ f(x) dx = ∫₀¹ f(s + (1 − t)/t) / t² dt
    private struct UpperTail<T>(T f, double origin) : IIntegrand where T : IIntegrand
    {
        private T f = f;

        // t² のアンダーフローを避けるため t で2回除算する（f が速く減衰すれば 0 になる）
        public double Evaluate(double t) => f.Evaluate(origin + (1 - t) / t) / t / t;
    }

    // ∫_(−∞)^s f(x) dx = ∫₀¹ f(s − (1 − t)/t) / t² dt
    private struct LowerTail<T>(T f, double origin) : IIntegrand where T : IIntegrand
    {
        private T f = f;

        public double Evaluate(double t) => f.Evaluate(origin - (1 - t) / t) / t / t;
    }

    // 誤差見積もりが最大の区間から順に二分割する大域的適応型求積
    private sealed class Adaptive<T>(T f, double lowerOrigin, double upperOrigin) where T : IIntegrand
    {
        private T finite = f;
        private LowerTail<T> lower = new(f, lowerOrigin);
        private UpperTail<T> upper = new(f, upperOrigin);

        private readonly PriorityQueue<Interval, double> heap = new();
        private readonly List<Interval> frozen = [];
        private CompensatedSum value;
        private CompensatedSum heapError;
        private CompensatedSum frozenError;
        private bool tooSmall;
        private long evaluations;

        public IntegrationResult Integrate(List<Piece> pieces, IntegrationOptions options)
        {
            foreach (var piece in pieces)
            {
                if (!Apply(piece.Kind, piece.T0, piece.T1))
                {
                    return NonFinite();
                }
            }

            while (true)
            {
                double target = Math.Max(options.AbsoluteTolerance, options.RelativeTolerance * Math.Abs(value.Value));
                if (heapError.Value + frozenError.Value <= target)
                {
                    // 加減算を重ねた補償和の誤差を確認するため、全区間の和を再計算する
                    Recompute();
                    target = Math.Max(options.AbsoluteTolerance, options.RelativeTolerance * Math.Abs(value.Value));
                    if (heapError.Value + frozenError.Value <= target)
                    {
                        return Result(IntegrationTermination.Tolerance);
                    }
                }

                // 分割しても減らない誤差だけで許容誤差を超えた場合は収束不能
                if (frozenError.Value > target || heap.Count == 0)
                {
                    return Result(tooSmall ? IntegrationTermination.IntervalTooSmall : IntegrationTermination.RoundoffLimited);
                }

                if (evaluations > options.MaxEvaluations - 2 * RuleEvaluations)
                {
                    return Result(IntegrationTermination.MaxEvaluations);
                }

                var worst = heap.Dequeue();
                heapError.Add(-worst.Error);
                double t0 = worst.T0, t1 = worst.T1;
                double middle = 0.5 * t0 + 0.5 * t1;
                double width = t1 - t0;
                if (!(middle > t0 && middle < t1) || width <= RelativeMinimumWidth * Math.Max(Math.Abs(t0), Math.Abs(t1)) || width <= AbsoluteMinimumWidth)
                {
                    tooSmall = true;
                    frozen.Add(worst);
                    frozenError.Add(worst.Error);
                    continue;
                }

                value.Add(-worst.Value);
                if (!Apply(worst.Kind, t0, middle) || !Apply(worst.Kind, middle, t1))
                {
                    return NonFinite();
                }
            }
        }

        private bool Apply(PieceKind kind, double t0, double t1)
        {
            evaluations += RuleEvaluations;
            RuleEstimate estimate;
            bool finiteValues;
            if (kind == PieceKind.LowerTail)
            {
                finiteValues = GaussKronrod.Apply(ref lower, t0, t1, out estimate);
            }
            else if (kind == PieceKind.UpperTail)
            {
                finiteValues = GaussKronrod.Apply(ref upper, t0, t1, out estimate);
            }
            else
            {
                finiteValues = GaussKronrod.Apply(ref finite, t0, t1, out estimate);
            }

            if (!finiteValues)
            {
                return false;
            }

            var interval = new Interval(t0, t1, kind, estimate.Value, estimate.Error);
            value.Add(interval.Value);
            if (estimate.AtRoundoff)
            {
                frozen.Add(interval);
                frozenError.Add(interval.Error);
            }
            else
            {
                heap.Enqueue(interval, -interval.Error);
                heapError.Add(interval.Error);
            }

            return true;
        }

        private void Recompute()
        {
            value = default;
            heapError = default;
            frozenError = default;
            foreach (var (interval, _) in heap.UnorderedItems)
            {
                value.Add(interval.Value);
                heapError.Add(interval.Error);
            }

            foreach (var interval in frozen)
            {
                value.Add(interval.Value);
                frozenError.Add(interval.Error);
            }
        }

        private IntegrationResult Result(IntegrationTermination termination)
        {
            Recompute();
            double total = value.Value;
            double error = heapError.Value + frozenError.Value;
            if (!double.IsFinite(total) || !double.IsFinite(error))
            {
                return NonFinite();
            }

            return new IntegrationResult(total, error, evaluations, termination);
        }

        private IntegrationResult NonFinite() =>
            new(double.NaN, double.NaN, evaluations, IntegrationTermination.NonFiniteValue);
    }
}
