using Tuna.LinearAlgebra;

namespace Tuna.Optimization.Proximal;

/// <summary>
/// Represents the least-squares objective <c>f(x) = ½‖Ax − b‖²</c>, usable as both a differentiable objective (for <see cref="Fista"/>) and a proximal operator (for <see cref="Admm"/>).
/// <para>最小二乗目的関数 <c>f(x) = ½‖Ax − b‖²</c> を表します。微分可能な目的関数（<see cref="Fista"/> の滑らかな項）としても、近接写像（<see cref="Admm"/> の項）としても利用できます。</para>
/// </summary>
/// <remarks>
/// <para>Copies <c>A</c> and <c>b</c> on construction. The proximal operator caches the Gram matrix and Cholesky factorization (applying the Woodbury identity when <c>m &lt; n</c>) and only re-factors when the step parameter changes. Instances are not thread-safe and must not be called concurrently.</para>
/// <para>構築時に <c>A</c> と <c>b</c> をコピーして保持します。近接写像ではグラム行列とコレスキー分解（<c>m &lt; n</c> の場合は Woodbury の公式を使用）をキャッシュし、ステップ幅が変化したときのみ再分解します。インスタンスはスレッドセーフではないため、複数スレッドから同時に呼び出さないでください。</para>
/// </remarks>
public sealed class LeastSquaresObjective : IDifferentiableObjectiveFunction, IProximalOperator
{
    private readonly Matrix<double> a;
    private readonly double[] b;
    private readonly double[] residual;
    private readonly double[] point;
    private readonly double[] gradient;
    private double[]? atb;
    private Matrix<double>? gram;          // AᵀA if m ≥ n, otherwise AAᵀ
    private Matrix<double>? shifted;
    private CholeskyDecomposition? factor;
    private double factorPenalty = double.NaN;

    /// <summary>
    /// Initializes a least-squares objective <c>½‖Ax − b‖²</c>.
    /// <para>最小二乗目的関数 <c>½‖Ax − b‖²</c> を初期化します。</para>
    /// </summary>
    /// <param name="a">The coefficient matrix <c>A</c> of size <c>m × n</c> (at least 1 × 1, finite). / 係数行列 <c>A</c>（<c>m × n</c>、1 × 1 以上の有限値）。</param>
    /// <param name="b">The right-hand side vector <c>b</c> of length <c>m</c> (finite). / 右辺ベクトル <c>b</c>（長さ <c>m</c>、有限値）。</param>
    /// <exception cref="ArgumentNullException"><paramref name="a"/> is <see langword="null"/>. / <paramref name="a"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException">The dimensions are empty or mismatched, or any element is non-finite. / 次元が空または一致しない、あるいは有限でない値を含む場合。</exception>
    public LeastSquaresObjective(Matrix<double> a, ReadOnlyVectorSpan<double> b)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a.Rows == 0 || a.Columns == 0)
        {
            throw new ArgumentException("The matrix must not be empty.", nameof(a));
        }

        if (b.Length != a.Rows)
        {
            throw new ArgumentException("The right-hand side length must equal the number of rows.", nameof(b));
        }

        this.a = a.Clone();
        this.b = b.ToArray();
        QuadraticChecks.Finite(this.a.AsSpan(), nameof(a));
        QuadraticChecks.Finite(this.b, nameof(b));
        residual = new double[a.Rows];
        point = new double[a.Columns];
        gradient = new double[a.Columns];
    }

    /// <summary>
    /// Gets the number of rows <c>m</c> of the matrix <c>A</c>.
    /// <para>行列 <c>A</c> の行数 <c>m</c> を取得します。</para>
    /// </summary>
    public int Rows => a.Rows;

    /// <summary>
    /// Gets the number of variables <c>n</c>.
    /// <para>変数の数 <c>n</c> を取得します。</para>
    /// </summary>
    public int Dimension => a.Columns;

    /// <summary>
    /// Evaluates <c>½‖Ax − b‖²</c> at <paramref name="x"/>.
    /// <para><paramref name="x"/> における <c>½‖Ax − b‖²</c> を評価します。</para>
    /// </summary>
    /// <param name="x">The point to evaluate (length <see cref="Dimension"/>). / 評価点（長さ <see cref="Dimension"/>）。</param>
    /// <returns>The least-squares objective value <c>½‖Ax − b‖²</c>. / 最小二乗目的関数値 <c>½‖Ax − b‖²</c>。</returns>
    /// <exception cref="ArgumentException">The length of <paramref name="x"/> does not match <see cref="Dimension"/>. / <paramref name="x"/> の長さが <see cref="Dimension"/> と一致しない場合。</exception>
    public double Evaluate(ReadOnlyVectorSpan<double> x)
    {
        Residual(x);
        double norm = Blas.Norm2(residual);
        return 0.5 * norm * norm;
    }

    /// <summary>
    /// Evaluates <c>½‖Ax − b‖²</c> and writes its gradient <c>Aᵀ(Ax − b)</c> to <paramref name="gradient"/>.
    /// <para><c>½‖Ax − b‖²</c> を評価し、勾配 <c>Aᵀ(Ax − b)</c> を <paramref name="gradient"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The point to evaluate (length <see cref="Dimension"/>). / 評価点（長さ <see cref="Dimension"/>）。</param>
    /// <param name="gradient">The destination span for the gradient (length <see cref="Dimension"/>). / 勾配の書き込み先（長さ <see cref="Dimension"/>）。</param>
    /// <returns>The least-squares objective value <c>½‖Ax − b‖²</c>. / 最小二乗目的関数値 <c>½‖Ax − b‖²</c>。</returns>
    /// <exception cref="ArgumentException">The length of <paramref name="x"/> or <paramref name="gradient"/> does not match <see cref="Dimension"/>. / <paramref name="x"/> または <paramref name="gradient"/> の長さが <see cref="Dimension"/> と一致しない場合。</exception>
    public double Evaluate(ReadOnlyVectorSpan<double> x, VectorSpan<double> gradient)
    {
        QuadraticChecks.Length(gradient.Length, Dimension, nameof(gradient));
        double value = Evaluate(x);
        QuadraticChecks.Multiply(a.T, residual, this.gradient);
        new ReadOnlyVectorSpan<double>(this.gradient).CopyTo(gradient);
        return value;
    }

    /// <inheritdoc/>
    public void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination)
    {
        ProximalChecks.Check(v, step, destination, Dimension);
        double rho = 1 / step;
        Prepare(rho);
        int m = Rows, n = Dimension;

        // w = Aᵀb + ρv
        for (int i = 0; i < n; i++)
        {
            point[i] = atb![i] + rho * v[i];
        }

        if (m >= n)
        {
            factor!.Solve(point, gradient);
        }
        else
        {
            // Woodbury identity: x = (w − Aᵀ(AAᵀ + ρI)⁻¹Aw) / ρ
            QuadraticChecks.Multiply(a, point, residual);
            factor!.Solve(residual, residual);
            QuadraticChecks.Multiply(a.T, residual, gradient);
            for (int i = 0; i < n; i++)
            {
                gradient[i] = (point[i] - gradient[i]) / rho;
            }
        }

        new ReadOnlyVectorSpan<double>(gradient).CopyTo(destination);
    }

    private void Residual(ReadOnlyVectorSpan<double> x)
    {
        QuadraticChecks.Length(x.Length, Dimension, nameof(x));
        x.CopyTo(point);
        QuadraticChecks.Multiply(a, point, residual);
        for (int i = 0; i < residual.Length; i++)
        {
            residual[i] -= b[i];
        }
    }

    private void Prepare(double rho)
    {
        if (gram is null)
        {
            gram = Rows >= Dimension ? XMath.Multiply(a.T, a) : XMath.Multiply(a, a.T);
            shifted = new Matrix<double>(gram.Rows, gram.Columns);
            atb = new double[Dimension];
            QuadraticChecks.Multiply(a.T, b, atb);
        }

        if (rho == factorPenalty)
        {
            return;
        }

        factor = QuadraticChecks.FactorShifted(gram, rho, shifted!);
        factorPenalty = rho;
    }
}

/// <summary>
/// Represents the convex quadratic objective <c>f(x) = ½xᵀPx + qᵀx + c</c> for symmetric positive semidefinite <c>P</c>, usable as both a differentiable objective and a proximal operator.
/// <para>対称半正定値行列 <c>P</c> に対する二次形式目的関数 <c>f(x) = ½xᵀPx + qᵀx + c</c> を表します。微分可能な目的関数としても近接写像としても利用できます。</para>
/// </summary>
/// <remarks>
/// <para><c>P</c> is symmetrized as <c>(P + Pᵀ) / 2</c> and copied on construction. The proximal operator caches the Cholesky factorization of <c>P + ρI</c> (<c>ρ = 1/t</c>) until the step changes. Instances are not thread-safe and must not be called concurrently.</para>
/// <para>構築時に <c>P</c> を <c>(P + Pᵀ) / 2</c> に対称化してコピー保持します。近接写像ではステップ幅が変わるまで <c>P + ρI</c>（<c>ρ = 1/t</c>）のコレスキー分解を再利用します。インスタンスはスレッドセーフではないため、複数スレッドから同時に呼び出さないでください。</para>
/// </remarks>
public sealed class QuadraticObjective : IDifferentiableObjectiveFunction, IProximalOperator
{
    private readonly Matrix<double> p;
    private readonly double[] q;
    private readonly double constant;
    private readonly double[] point;
    private readonly double[] product;
    private readonly Matrix<double> shifted;
    private CholeskyDecomposition? factor;
    private double factorPenalty = double.NaN;

    /// <summary>
    /// Initializes a quadratic objective <c>½xᵀPx + qᵀx + c</c>.
    /// <para>二次形式目的関数 <c>½xᵀPx + qᵀx + c</c> を初期化します。</para>
    /// </summary>
    /// <param name="p">The symmetric positive semidefinite matrix <c>P</c> of size <c>n × n</c> (at least 1 × 1, finite). / 対称半正定値行列 <c>P</c>（<c>n × n</c>、1 × 1 以上の有限値）。</param>
    /// <param name="q">The linear coefficient vector <c>q</c> of length <c>n</c> (finite). / 1次の係数ベクトル <c>q</c>（長さ <c>n</c>、有限値）。</param>
    /// <param name="constant">The constant term <c>c</c> (finite; default is 0). / 定数項 <c>c</c>（有限値、既定値は 0）。</param>
    /// <exception cref="ArgumentNullException"><paramref name="p"/> is <see langword="null"/>. / <paramref name="p"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException">The dimensions are empty, non-square, or mismatched, or any value is non-finite. / 次元が空・非正方・不一致、または有限でない値を含む場合。</exception>
    public QuadraticObjective(Matrix<double> p, ReadOnlyVectorSpan<double> q, double constant = 0)
    {
        this.p = QuadraticChecks.Symmetrized(p, nameof(p));
        if (q.Length != this.p.Rows)
        {
            throw new ArgumentException("The linear term length must equal the order of P.", nameof(q));
        }

        this.q = q.ToArray();
        QuadraticChecks.Finite(this.q, nameof(q));
        if (!double.IsFinite(constant))
        {
            throw new ArgumentException("The constant must be finite.", nameof(constant));
        }

        this.constant = constant;
        point = new double[this.p.Rows];
        product = new double[this.p.Rows];
        shifted = new Matrix<double>(this.p.Rows, this.p.Rows);
    }

    /// <summary>
    /// Gets the number of variables <c>n</c>.
    /// <para>変数の数 <c>n</c> を取得します。</para>
    /// </summary>
    public int Dimension => p.Rows;

    /// <summary>
    /// Evaluates <c>½xᵀPx + qᵀx + c</c> at <paramref name="x"/>.
    /// <para><paramref name="x"/> における <c>½xᵀPx + qᵀx + c</c> を評価します。</para>
    /// </summary>
    /// <param name="x">The point to evaluate (length <see cref="Dimension"/>). / 評価点（長さ <see cref="Dimension"/>）。</param>
    /// <returns>The quadratic objective value <c>½xᵀPx + qᵀx + c</c>. / 二次形式の目的関数値 <c>½xᵀPx + qᵀx + c</c>。</returns>
    /// <exception cref="ArgumentException">The length of <paramref name="x"/> does not match <see cref="Dimension"/>. / <paramref name="x"/> の長さが <see cref="Dimension"/> と一致しない場合。</exception>
    public double Evaluate(ReadOnlyVectorSpan<double> x)
    {
        Product(x);
        double value = 0;
        for (int i = 0; i < point.Length; i++)
        {
            value += point[i] * (0.5 * product[i] + q[i]);
        }

        return value + constant;
    }

    /// <summary>
    /// Evaluates <c>½xᵀPx + qᵀx + c</c> and writes its gradient <c>Px + q</c> to <paramref name="gradient"/>.
    /// <para><c>½xᵀPx + qᵀx + c</c> を評価し、勾配 <c>Px + q</c> を <paramref name="gradient"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The point to evaluate (length <see cref="Dimension"/>). / 評価点（長さ <see cref="Dimension"/>）。</param>
    /// <param name="gradient">The destination span for the gradient (length <see cref="Dimension"/>). / 勾配の書き込み先（長さ <see cref="Dimension"/>）。</param>
    /// <returns>The quadratic objective value <c>½xᵀPx + qᵀx + c</c>. / 二次形式の目的関数値 <c>½xᵀPx + qᵀx + c</c>。</returns>
    /// <exception cref="ArgumentException">The length of <paramref name="x"/> or <paramref name="gradient"/> does not match <see cref="Dimension"/>. / <paramref name="x"/> または <paramref name="gradient"/> の長さが <see cref="Dimension"/> と一致しない場合。</exception>
    public double Evaluate(ReadOnlyVectorSpan<double> x, VectorSpan<double> gradient)
    {
        QuadraticChecks.Length(gradient.Length, Dimension, nameof(gradient));
        double value = Evaluate(x);
        for (int i = 0; i < product.Length; i++)
        {
            gradient[i] = product[i] + q[i];
        }

        return value;
    }

    /// <exception cref="InvalidOperationException"><c>P + (1/step)I</c> is not positive definite (<c>P</c> is not positive semidefinite). / <c>P + (1/step)I</c> が正定値でない（<c>P</c> が半正定値でない）場合。</exception>
    /// <inheritdoc/>
    public void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination)
    {
        ProximalChecks.Check(v, step, destination, Dimension);
        double rho = 1 / step;
        if (rho != factorPenalty)
        {
            factor = QuadraticChecks.FactorShifted(p, rho, shifted);
            factorPenalty = rho;
        }

        for (int i = 0; i < point.Length; i++)
        {
            point[i] = rho * v[i] - q[i];
        }

        factor!.Solve(point, product);
        new ReadOnlyVectorSpan<double>(product).CopyTo(destination);
    }

    private void Product(ReadOnlyVectorSpan<double> x)
    {
        QuadraticChecks.Length(x.Length, Dimension, nameof(x));
        x.CopyTo(point);
        QuadraticChecks.Multiply(p, point, product);
    }
}

internal static class QuadraticChecks
{
    // result = M x（配列どうし）
    public static void Multiply(ReadOnlyMatrixSpan<double> m, double[] x, double[] result) =>
        XMath.Multiply(m, new ReadOnlyVectorSpan<double>(x), new VectorSpan<double>(result));

    public static void Length(int length, int expected, string name)
    {
        if (length != expected)
        {
            throw new ArgumentException($"The vector length must be {expected}.", name);
        }
    }

    public static void Finite(ReadOnlySpan<double> values, string name)
    {
        foreach (double value in values)
        {
            if (!double.IsFinite(value))
            {
                throw new ArgumentException("The values must be finite.", name);
            }
        }
    }

    public static Matrix<double> Symmetrized(Matrix<double> p, string name)
    {
        ArgumentNullException.ThrowIfNull(p, name);
        if (!p.IsSquare || p.Rows == 0)
        {
            throw new ArgumentException("The matrix must be square and not empty.", name);
        }

        var result = new Matrix<double>(p.Rows, p.Rows);
        for (int j = 0; j < p.Rows; j++)
        {
            for (int i = 0; i < p.Rows; i++)
            {
                result[i, j] = 0.5 * (p[i, j] + p[j, i]);
            }
        }

        Finite(result.AsSpan(), name);
        return result;
    }

    // (G + ρI) のコレスキー分解を destination に作る
    public static CholeskyDecomposition FactorShifted(Matrix<double> g, double rho, Matrix<double> destination)
    {
        g.AsSpan().CopyTo(destination.AsSpan());
        for (int i = 0; i < g.Rows; i++)
        {
            destination[i, i] += rho;
        }

        if (!CholeskyDecomposition.TryCompute(destination, destination, out var factor))
        {
            throw new InvalidOperationException("The shifted matrix is not positive definite; the quadratic term must be positive semidefinite.");
        }

        return factor;
    }
}
