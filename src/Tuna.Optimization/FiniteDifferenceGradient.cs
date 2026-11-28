using Tuna.LinearAlgebra;

namespace Tuna.Optimization;

// Step sizes follow Nocedal & Wright (2006), §8.1: ε^(1/3)·max(1, |xᵢ|) for central differences
// and ε^(1/2)·max(1, |xᵢ|) for one-sided differences.
/// <summary>
/// Approximates the gradient of a value-only objective function using finite differences.
/// <para>値のみを返す目的関数に対し、有限差分で勾配を近似するラッパーです。</para>
/// </summary>
/// <remarks>
/// <para>Uses central differences by default, switching to one-sided differences (or the secant across the bounds for narrow boxes) whenever a central step would leave the specified bounds. The wrapped objective is never evaluated outside the bounds. Call counts are tracked in <see cref="FunctionEvaluations"/>. Instances are not thread-safe and must not be called concurrently.</para>
/// <para>既定では中心差分を使用し、範囲制約を指定した場合に中心差分の評価点が範囲外に出る成分では片側差分（区間幅が刻みより狭い場合は両端の割線）に切り替えます。範囲外では評価しません。評価回数は <see cref="FunctionEvaluations"/> で取得できます。インスタンスはスレッドセーフではないため、複数スレッドから同時に呼び出さないでください。</para>
/// </remarks>
public sealed class FiniteDifferenceGradient : IDifferentiableObjectiveFunction
{
    private const double OneSidedScale = NumericConstants.SqrtMachineEpsilon;
    private static readonly double CentralScale = Math.Cbrt(NumericConstants.MachineEpsilon);

    private readonly Func<ReadOnlyVectorSpan<double>, double>? function;
    private readonly IObjectiveFunction? objective;
    private readonly double[]? lower;
    private readonly double[]? upper;
    private readonly double[] work;

    /// <summary>
    /// Initializes a new instance that approximates the gradient of a delegate objective function.
    /// <para>デリゲートで与えた目的関数の勾配を有限差分で近似するインスタンスを初期化します。</para>
    /// </summary>
    /// <param name="function">The objective function returning scalar values. / 目的関数値を返すデリゲート。</param>
    /// <param name="dimension">The number of variables (at least 1). / 変数の数（1 以上）。</param>
    /// <param name="lowerBounds">Optional lower bounds that finite-difference evaluations must not violate (may contain <see cref="double.NegativeInfinity"/>). / 有限差分の評価点が下回ってはならない下限（省略時は無制限。<see cref="double.NegativeInfinity"/> も指定可能）。</param>
    /// <param name="upperBounds">Optional upper bounds that finite-difference evaluations must not violate (may contain <see cref="double.PositiveInfinity"/>). / 有限差分の評価点が上回ってはならない上限（省略時は無制限。<see cref="double.PositiveInfinity"/> も指定可能）。</param>
    /// <exception cref="ArgumentNullException"><paramref name="function"/> is <see langword="null"/>. / <paramref name="function"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="dimension"/> is less than 1. / <paramref name="dimension"/> が 1 未満の場合。</exception>
    /// <exception cref="ArgumentException">The bounds have an invalid length, contain <see cref="double.NaN"/>, or define an empty box. / 範囲制約の長さが次元と一致しない、<see cref="double.NaN"/> を含む、または下限が上限を超えている場合。</exception>
    public FiniteDifferenceGradient(Func<ReadOnlyVectorSpan<double>, double> function, int dimension,
        ReadOnlySpan<double> lowerBounds = default, ReadOnlySpan<double> upperBounds = default)
        : this(dimension, lowerBounds, upperBounds)
    {
        ArgumentNullException.ThrowIfNull(function);
        this.function = function;
    }

    /// <summary>
    /// Initializes a new instance that approximates the gradient of an <see cref="IObjectiveFunction"/>.
    /// <para><see cref="IObjectiveFunction"/> の勾配を有限差分で近似するインスタンスを初期化します。</para>
    /// </summary>
    /// <param name="objective">The objective function to wrap (boxed if a value type). / ラップする目的関数（値型の場合はコピーされます）。</param>
    /// <param name="dimension">The number of variables (at least 1). / 変数の数（1 以上）。</param>
    /// <param name="lowerBounds">Optional lower bounds that finite-difference evaluations must not violate (may contain <see cref="double.NegativeInfinity"/>). / 有限差分の評価点が下回ってはならない下限（省略時は無制限。<see cref="double.NegativeInfinity"/> も指定可能）。</param>
    /// <param name="upperBounds">Optional upper bounds that finite-difference evaluations must not violate (may contain <see cref="double.PositiveInfinity"/>). / 有限差分の評価点が上回ってはならない上限（省略時は無制限。<see cref="double.PositiveInfinity"/> も指定可能）。</param>
    /// <exception cref="ArgumentNullException"><paramref name="objective"/> is <see langword="null"/>. / <paramref name="objective"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="dimension"/> is less than 1. / <paramref name="dimension"/> が 1 未満の場合。</exception>
    /// <exception cref="ArgumentException">The bounds have an invalid length, contain <see cref="double.NaN"/>, or define an empty box. / 範囲制約の長さが次元と一致しない、<see cref="double.NaN"/> を含む、または下限が上限を超えている場合。</exception>
    public FiniteDifferenceGradient(IObjectiveFunction objective, int dimension,
        ReadOnlySpan<double> lowerBounds = default, ReadOnlySpan<double> upperBounds = default)
        : this(dimension, lowerBounds, upperBounds)
    {
        ArgumentNullException.ThrowIfNull(objective);
        this.objective = objective;
    }

    private FiniteDifferenceGradient(int dimension, ReadOnlySpan<double> lowerBounds, ReadOnlySpan<double> upperBounds)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(dimension, 1);
        Bounds.Validate(lowerBounds, upperBounds, dimension, allowEmpty: true, out lower, out upper);
        work = new double[dimension];
    }

    /// <summary>
    /// Gets the number of variables.
    /// <para>変数の数を取得します。</para>
    /// </summary>
    public int Dimension => work.Length;

    /// <summary>
    /// Gets the total number of evaluations performed on the wrapped objective, including value-only evaluations.
    /// <para>ラップした目的関数の評価回数（値のみの評価を含む）を取得します。</para>
    /// </summary>
    public long FunctionEvaluations { get; private set; }

    /// <summary>
    /// Evaluates the wrapped objective function at the specified point.
    /// <para>指定した点でラップした目的関数を評価します。</para>
    /// </summary>
    /// <param name="x">The point to evaluate. / 評価点。</param>
    /// <returns>The objective value at <paramref name="x"/>. / <paramref name="x"/> での目的関数値。</returns>
    /// <exception cref="ArgumentException">The length of <paramref name="x"/> does not match <see cref="Dimension"/>. / <paramref name="x"/> の長さが <see cref="Dimension"/> と一致しない場合。</exception>
    public double Evaluate(ReadOnlyVectorSpan<double> x)
    {
        CheckLength(x.Length);
        return Call(x);
    }

    /// <summary>
    /// Evaluates the wrapped objective function at the specified point and writes its finite-difference gradient approximation.
    /// <para>指定した点で目的関数を評価し、有限差分で近似した勾配を書き込みます。</para>
    /// </summary>
    /// <param name="x">The point to evaluate. / 評価点。</param>
    /// <param name="gradient">The destination span for the approximated gradient, with length equal to <see cref="Dimension"/>. / 近似勾配の書き込み先。長さは <see cref="Dimension"/> と同じ。</param>
    /// <returns>The objective value at <paramref name="x"/>. / <paramref name="x"/> での目的関数値。</returns>
    /// <exception cref="ArgumentException">The length of <paramref name="x"/> or <paramref name="gradient"/> does not match <see cref="Dimension"/>. / <paramref name="x"/> または <paramref name="gradient"/> の長さが <see cref="Dimension"/> と一致しない場合。</exception>
    public double Evaluate(ReadOnlyVectorSpan<double> x, VectorSpan<double> gradient)
    {
        CheckLength(x.Length);
        CheckLength(gradient.Length);
        x.CopyTo(work);
        double value = Call(work);
        for (int i = 0; i < work.Length; i++)
        {
            double xi = work[i];
            double lo = lower?[i] ?? double.NegativeInfinity, hi = upper?[i] ?? double.PositiveInfinity;
            double scale = Math.Max(1.0, Math.Abs(xi));
            double hc = Representable(xi, CentralScale * scale);
            double hs = Representable(xi, OneSidedScale * scale);
            double slope;
            if (xi - hc >= lo && xi + hc <= hi)
            {
                slope = (At(i, xi + hc) - At(i, xi - hc)) / (2 * hc);
            }
            else if (xi + hs <= hi)
            {
                slope = (At(i, xi + hs) - value) / hs;
            }
            else if (xi - hs >= lo)
            {
                slope = (value - At(i, xi - hs)) / hs;
            }
            else if (hi > lo)
            {
                // 範囲が刻みより狭い: 両端の割線
                slope = (At(i, hi) - At(i, lo)) / (hi - lo);
            }
            else
            {
                slope = 0;   // 固定された変数
            }

            work[i] = xi;
            gradient[i] = slope;
        }

        return value;
    }

    private double At(int i, double xi)
    {
        work[i] = xi;
        return Call(work);
    }

    private double Call(ReadOnlyVectorSpan<double> x)
    {
        FunctionEvaluations++;
        return function is not null ? function(x) : objective!.Evaluate(x);
    }

    private void CheckLength(int length)
    {
        if (length != work.Length)
        {
            throw new ArgumentException($"The vector length must be {work.Length}.");
        }
    }

    // 刻みの丸め誤差を差分に入れないよう、実際に評価する点 x + h との差 (x + h) − x で割る（この差は正確に表せる）
    private static double Representable(double x, double h) => (x + h) - x;
}
