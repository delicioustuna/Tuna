using Tuna.LinearAlgebra;

namespace Tuna.Optimization.Proximal;

/// <summary>
/// Configures options for <see cref="Admm"/>.
/// <para><see cref="Admm"/> の設定オプションを表します。</para>
/// </summary>
/// <remarks>
/// <para>Option values are validated by the <see cref="Admm"/> constructor.</para>
/// <para>各設定値の妥当性は <see cref="Admm"/> のコンストラクターで検証されます。</para>
/// </remarks>
public sealed class AdmmOptions
{
    /// <summary>
    /// Gets the initial augmented Lagrangian penalty parameter ρ.
    /// <para>拡張ラグランジュ法の罰則係数 ρ の初期値を取得します。</para>
    /// </summary>
    /// <value>A positive finite number. The default is 1. / 正の有限値。既定値は 1。</value>
    public double Penalty { get; init; } = 1;

    /// <summary>
    /// Gets the absolute residual stopping tolerance <c>ε_abs</c>.
    /// <para>残差停止条件の絶対許容誤差 <c>ε_abs</c> を取得します。</para>
    /// </summary>
    /// <value>A non-negative finite number. The default is <c>1e-6</c>. / 0 以上の有限値。既定値は <c>1e-6</c>。</value>
    public double AbsoluteTolerance { get; init; } = 1e-6;

    /// <summary>
    /// Gets the relative residual stopping tolerance <c>ε_rel</c>.
    /// <para>残差停止条件の相対許容誤差 <c>ε_rel</c> を取得します。</para>
    /// </summary>
    /// <value>A non-negative finite number. The default is <c>1e-4</c>. / 0 以上の有限値。既定値は <c>1e-4</c>。</value>
    public double RelativeTolerance { get; init; } = 1e-4;

    /// <summary>
    /// Gets the over-relaxation parameter α.
    /// <para>過緩和係数 α を取得します。</para>
    /// </summary>
    /// <value>A value strictly between 0 and 2. The default is 1 (no over-relaxation); values in 1.5–1.8 often accelerate convergence. / 0 より大きく 2 未満の値。既定値は 1（過緩和なし）。1.5〜1.8 で収束が速まることが多くあります。</value>
    public double OverRelaxation { get; init; } = 1;

    /// <summary>
    /// Gets a value indicating whether ρ is adaptively adjusted based on the primal-to-dual residual ratio.
    /// <para>主残差と双対残差の比に応じて罰則係数 ρ を適応的に調整するかどうかを取得します。</para>
    /// </summary>
    /// <value><see langword="true"/> (the default) to enable residual balancing; otherwise, <see langword="false"/>. / 残差バランシングを有効にする場合は <see langword="true"/>（既定値）、無効にする場合は <see langword="false"/>。</value>
    public bool AdaptivePenalty { get; init; } = true;

    /// <summary>
    /// Gets the residual ratio threshold μ triggering a ρ update.
    /// <para>罰則係数 ρ を更新する残差比のしきい値 μ を取得します。</para>
    /// </summary>
    /// <value>A value strictly greater than 1. The default is 10. / 1 より大きい値。既定値は 10。</value>
    public double PenaltyBalance { get; init; } = 10;

    /// <summary>
    /// Gets the multiplicative scaling factor τ applied when adjusting ρ.
    /// <para>罰則係数 ρ の調整時に掛けたり割ったりする倍率 τ を取得します。</para>
    /// </summary>
    /// <value>A finite value strictly greater than 1. The default is 2. / 1 より大きい有限値。既定値は 2。</value>
    public double PenaltyFactor { get; init; } = 2;

    /// <summary>
    /// Gets the maximum number of adaptive ρ updates allowed.
    /// <para>罰則係数 ρ の適応更新回数の上限を取得します。</para>
    /// </summary>
    /// <value>A non-negative integer. The default is 50; bounding the number of updates guarantees standard ADMM convergence. / 0 以上の整数。既定値は 50（更新回数を有限に制限することで収束保証を保ちます）。</value>
    public int MaxPenaltyUpdates { get; init; } = 50;

    /// <summary>
    /// Gets the maximum number of iterations.
    /// <para>反復回数の上限を取得します。</para>
    /// </summary>
    /// <value>An integer greater than or equal to 1. The default is 10000. / 1 以上の整数。既定値は 10000。</value>
    public int MaxIterations { get; init; } = 10000;
}

/// <summary>
/// Represents the result of an <see cref="Admm"/> minimization run.
/// <para><see cref="Admm"/> による最小化の実行結果を表します。</para>
/// </summary>
/// <param name="X">A copy of the final primal iterate <c>x</c>. / 最終主変数 <c>x</c> のコピー。</param>
/// <param name="Z">A copy of the final split variable <c>z</c> (lies in the domain of <c>g</c> as the output of its proximal operator). / 最終分割変数 <c>z</c> のコピー（<c>g</c> の近接写像の出力のため <c>g</c> の定義域内に属します）。</param>
/// <param name="ScaledDual">A copy of the final scaled dual variable <c>u = y / ρ</c> (Lagrange multiplier is <c>ρu</c>). / 最終スケール化双対変数 <c>u = y / ρ</c> のコピー（ラグランジュ乗数は <c>ρu</c>）。</param>
/// <param name="Value">The composite objective value <c>f(x) + g(z)</c>. / 複合目的関数値 <c>f(x) + g(z)</c>。</param>
/// <param name="PrimalResidual">The Euclidean norm of the primal residual <c>‖Dx − z‖₂</c> (or <c>‖x − z‖₂</c> in consensus form). / 主残差の L2 ノルム <c>‖Dx − z‖₂</c>（合意形式では <c>‖x − z‖₂</c>）。</param>
/// <param name="DualResidual">The Euclidean norm of the dual residual <c>‖ρDᵀ(z − z_old)‖₂</c>. / 双対残差の L2 ノルム <c>‖ρDᵀ(z − z_old)‖₂</c>。</param>
/// <param name="Penalty">The final penalty parameter ρ. / 最終的な罰則係数 ρ。</param>
/// <param name="Iterations">The number of completed iterations. / 完了した反復回数。</param>
/// <param name="Termination">The termination reason (<see cref="MinimizationTermination.ResidualTolerance"/>, <see cref="MinimizationTermination.MaxIterations"/>, or <see cref="MinimizationTermination.NonFiniteValue"/>). / 終了理由（<see cref="MinimizationTermination.ResidualTolerance"/>、<see cref="MinimizationTermination.MaxIterations"/>、または <see cref="MinimizationTermination.NonFiniteValue"/>）。</param>
public sealed record AdmmResult(
    Vector<double> X, Vector<double> Z, Vector<double> ScaledDual, double Value,
    double PrimalResidual, double DualResidual, double Penalty, int Iterations, MinimizationTermination Termination)
{
    /// <summary>
    /// Gets a value indicating whether ADMM terminated by satisfying the residual tolerances.
    /// <para>残差の許容誤差を満たして収束したかどうかを取得します。</para>
    /// </summary>
    /// <value><see langword="true"/> if <see cref="Termination"/> is <see cref="MinimizationTermination.ResidualTolerance"/>; otherwise, <see langword="false"/>. / <see cref="Termination"/> が <see cref="MinimizationTermination.ResidualTolerance"/> の場合は <see langword="true"/>、それ以外は <see langword="false"/>。</value>
    public bool Converged => Termination == MinimizationTermination.ResidualTolerance;
}

/// <summary>
/// Represents the <c>x</c>-update subproblem <c>argmin_x [f(x) + (ρ/2)‖Dx − c‖²]</c> and linear coupling <c>D</c> (<c>p × n</c>) for splitting-form ADMM <c>f(x) + g(Dx)</c>.
/// <para>分割形式 ADMM <c>f(x) + g(Dx)</c> における <c>x</c> 更新部分問題 <c>argmin_x [f(x) + (ρ/2)‖Dx − c‖²]</c> と線形写像 <c>D</c>（<c>p × n</c>）を表します。</para>
/// </summary>
/// <remarks>
/// <para>See <see cref="QuadraticAdmmSubproblem"/> for the built-in quadratic implementation.</para>
/// <para>二次形式向けの既製実装として <see cref="QuadraticAdmmSubproblem"/> を提供しています。</para>
/// </remarks>
public interface IAdmmSubproblem
{
    /// <summary>
    /// Gets the dimension <c>n</c> of the primal variable <c>x</c>.
    /// <para>主変数 <c>x</c> の次元 <c>n</c> を取得します。</para>
    /// </summary>
    int Dimension { get; }

    /// <summary>
    /// Gets the dimension <c>p</c> of <c>z = Dx</c> (the number of rows of <c>D</c>).
    /// <para>分割変数 <c>z = Dx</c> の次元 <c>p</c>（<c>D</c> の行数）を取得します。</para>
    /// </summary>
    int ConstraintCount { get; }

    /// <summary>
    /// Evaluates <c>f(x)</c>.
    /// <para><c>f(x)</c> を評価します。</para>
    /// </summary>
    /// <param name="x">The primal vector of length <see cref="Dimension"/>. / 長さ <see cref="Dimension"/> の主変数ベクトル。</param>
    /// <returns>The value <c>f(x)</c>. / 目的関数値 <c>f(x)</c>。</returns>
    double Evaluate(ReadOnlyVectorSpan<double> x);

    /// <summary>
    /// Computes <c>result = Dx</c>.
    /// <para><c>result = Dx</c> を計算します。</para>
    /// </summary>
    /// <param name="x">The input vector of length <see cref="Dimension"/>. / 長さ <see cref="Dimension"/> の入力ベクトル。</param>
    /// <param name="result">The destination span of length <see cref="ConstraintCount"/>. / 長さ <see cref="ConstraintCount"/> の出力先。</param>
    void Multiply(ReadOnlyVectorSpan<double> x, VectorSpan<double> result);

    /// <summary>
    /// Computes <c>result = Dᵀy</c>.
    /// <para><c>result = Dᵀy</c> を計算します。</para>
    /// </summary>
    /// <param name="y">The input vector of length <see cref="ConstraintCount"/>. / 長さ <see cref="ConstraintCount"/> の入力ベクトル。</param>
    /// <param name="result">The destination span of length <see cref="Dimension"/>. / 長さ <see cref="Dimension"/> の出力先。</param>
    void MultiplyTransposed(ReadOnlyVectorSpan<double> y, VectorSpan<double> result);

    /// <summary>
    /// Solves <c>argmin_x [f(x) + (ρ/2)‖Dx − target‖²]</c> and writes the solution to <paramref name="x"/>.
    /// <para><c>argmin_x [f(x) + (ρ/2)‖Dx − target‖²]</c> を解き、解を <paramref name="x"/> に書き込みます。</para>
    /// </summary>
    /// <param name="target">The target vector <c>c</c> of length <see cref="ConstraintCount"/>. / 長さ <see cref="ConstraintCount"/> の目標ベクトル <c>c</c>。</param>
    /// <param name="penalty">The positive penalty parameter ρ. Implementations should cache factorizations until <paramref name="penalty"/> changes. / 正の罰則係数 ρ。<paramref name="penalty"/> が変化するまで分解を再利用する実装が推奨されます。</param>
    /// <param name="x">The destination span of length <see cref="Dimension"/>; contains the previous iterate on entry for warm-starting iterative solvers. / 長さ <see cref="Dimension"/> の出力先。呼び出し時は直前の反復値を保持しており、反復法の初期値として利用できます。</param>
    void Solve(ReadOnlyVectorSpan<double> target, double penalty, VectorSpan<double> x);
}

/// <summary>
/// Represents a splitting-form ADMM subproblem for the quadratic term <c>f(x) = ½xᵀPx + qᵀx + c</c>.
/// <para>二次形式 <c>f(x) = ½xᵀPx + qᵀx + c</c> に対する分割形式 ADMM の部分問題を表します。</para>
/// </summary>
/// <remarks>
/// <para>Solves <c>(P + ρDᵀD)x = ρDᵀc − q</c> using a cached Cholesky factorization that is recomputed only when ρ changes. <c>P</c> is symmetrized as <c>(P + Pᵀ) / 2</c>, and both <c>P</c> and <c>D</c> are copied on construction. <c>P + ρDᵀD</c> must be positive definite.</para>
/// <para><c>(P + ρDᵀD)x = ρDᵀc − q</c> をコレスキー分解で解き、ρ が変化したときのみ再分解します。構築時に <c>P</c> を <c>(P + Pᵀ) / 2</c> に対称化し、<c>P</c> と <c>D</c> をコピー保持します。<c>P + ρDᵀD</c> は正定値である必要があります。</para>
/// </remarks>
public sealed class QuadraticAdmmSubproblem : IAdmmSubproblem
{
    private readonly Matrix<double> p;
    private readonly double[] q;
    private readonly double constant;
    private readonly Matrix<double> d;
    private readonly Matrix<double> dtd;
    private readonly Matrix<double> shifted;
    private readonly double[] point;
    private readonly double[] product;
    private readonly double[] constraint;
    private CholeskyDecomposition? factor;
    private double factorPenalty = double.NaN;

    /// <summary>
    /// Initializes a quadratic ADMM subproblem for <c>½xᵀPx + qᵀx + c</c> with linear operator <c>D</c>.
    /// <para>二次形式 <c>½xᵀPx + qᵀx + c</c> と線形写像 <c>D</c> から ADMM 部分問題を初期化します。</para>
    /// </summary>
    /// <param name="p">The symmetric positive semidefinite matrix <c>P</c> (<c>n × n</c>). / 対称半正定値行列 <c>P</c>（<c>n × n</c>）。</param>
    /// <param name="q">The linear coefficient vector <c>q</c> (length <c>n</c>). / 1次の係数ベクトル <c>q</c>（長さ <c>n</c>）。</param>
    /// <param name="d">The linear coupling matrix <c>D</c> (<c>p × n</c>, at least one row). / 線形結合行列 <c>D</c>（<c>p × n</c>、1行以上）。</param>
    /// <param name="constant">The constant term <c>c</c> (finite; default is 0). / 定数項 <c>c</c>（有限値、既定値は 0）。</param>
    /// <exception cref="ArgumentNullException"><paramref name="p"/> or <paramref name="d"/> is <see langword="null"/>. / <paramref name="p"/> または <paramref name="d"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException">Dimensions are empty or mismatched, or any value is non-finite. / 次元が空または不一致、あるいは有限でない値を含む場合。</exception>
    public QuadraticAdmmSubproblem(Matrix<double> p, ReadOnlyVectorSpan<double> q, Matrix<double> d, double constant = 0)
    {
        this.p = QuadraticChecks.Symmetrized(p, nameof(p));
        int n = this.p.Rows;
        if (q.Length != n)
        {
            throw new ArgumentException("The linear term length must equal the order of P.", nameof(q));
        }

        ArgumentNullException.ThrowIfNull(d);
        if (d.Columns != n || d.Rows == 0)
        {
            throw new ArgumentException("D must have as many columns as P and at least one row.", nameof(d));
        }

        if (!double.IsFinite(constant))
        {
            throw new ArgumentException("The constant must be finite.", nameof(constant));
        }

        this.q = q.ToArray();
        QuadraticChecks.Finite(this.q, nameof(q));
        this.d = d.Clone();
        QuadraticChecks.Finite(this.d.AsSpan(), nameof(d));
        this.constant = constant;
        dtd = XMath.Multiply(this.d.T, this.d);
        shifted = new Matrix<double>(n, n);
        point = new double[n];
        product = new double[n];
        constraint = new double[d.Rows];
    }

    /// <summary>
    /// Creates a quadratic ADMM subproblem for the least-squares term <c>½‖Ax − b‖²</c>.
    /// <para>最小二乗項 <c>½‖Ax − b‖²</c> に対応する ADMM 部分問題（<c>P = AᵀA</c>、<c>q = −Aᵀb</c>、<c>c = ½‖b‖²</c>）を生成します。</para>
    /// </summary>
    /// <param name="a">The coefficient matrix <c>A</c> (<c>m × n</c>). / 係数行列 <c>A</c>（<c>m × n</c>）。</param>
    /// <param name="b">The right-hand side vector <c>b</c> (length <c>m</c>). / 右辺ベクトル <c>b</c>（長さ <c>m</c>）。</param>
    /// <param name="d">The linear coupling matrix <c>D</c> (<c>p × n</c>). / 線形結合行列 <c>D</c>（<c>p × n</c>）。</param>
    /// <returns>A <see cref="QuadraticAdmmSubproblem"/> representing <c>½‖Ax − b‖²</c>. / <c>½‖Ax − b‖²</c> を表す <see cref="QuadraticAdmmSubproblem"/>。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="d"/> is <see langword="null"/>. / <paramref name="a"/> または <paramref name="d"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException">Dimensions are empty or mismatched, or any value is non-finite. / 次元が空または不一致、あるいは有限でない値を含む場合。</exception>
    public static QuadraticAdmmSubproblem LeastSquares(Matrix<double> a, ReadOnlyVectorSpan<double> b, Matrix<double> d)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (b.Length != a.Rows)
        {
            throw new ArgumentException("The right-hand side length must equal the number of rows.", nameof(b));
        }

        var rhs = b.ToArray();
        var q = new double[a.Columns];
        QuadraticChecks.Multiply(a.T, rhs, q);
        for (int i = 0; i < q.Length; i++)
        {
            q[i] = -q[i];
        }

        double norm = Blas.Norm2(rhs);
        return new QuadraticAdmmSubproblem(XMath.Multiply(a.T, a), q, d, 0.5 * norm * norm);
    }

    /// <inheritdoc/>
    public int Dimension => p.Rows;

    /// <inheritdoc/>
    public int ConstraintCount => d.Rows;

    /// <inheritdoc/>
    public double Evaluate(ReadOnlyVectorSpan<double> x)
    {
        QuadraticChecks.Length(x.Length, Dimension, nameof(x));
        x.CopyTo(point);
        QuadraticChecks.Multiply(p, point, product);
        double value = 0;
        for (int i = 0; i < point.Length; i++)
        {
            value += point[i] * (0.5 * product[i] + q[i]);
        }

        return value + constant;
    }

    /// <inheritdoc/>
    public void Multiply(ReadOnlyVectorSpan<double> x, VectorSpan<double> result) => XMath.Multiply(d, x, result);

    /// <inheritdoc/>
    public void MultiplyTransposed(ReadOnlyVectorSpan<double> y, VectorSpan<double> result) => XMath.Multiply(d.T, y, result);

    /// <exception cref="InvalidOperationException"><c>P + ρDᵀD</c> is not positive definite. / <c>P + ρDᵀD</c> が正定値でない場合。</exception>
    /// <inheritdoc/>
    public void Solve(ReadOnlyVectorSpan<double> target, double penalty, VectorSpan<double> x)
    {
        if (penalty != factorPenalty)
        {
            dtd.AsSpan().CopyTo(shifted.AsSpan());
            XMath.InPlace.Multiply(shifted, penalty);
            var data = shifted.AsSpan();
            var pData = p.AsSpan();
            for (int i = 0; i < data.Length; i++)
            {
                data[i] += pData[i];
            }

            if (!CholeskyDecomposition.TryCompute(shifted, shifted, out var computed))
            {
                factorPenalty = double.NaN;
                throw new InvalidOperationException("P + ρDᵀD is not positive definite.");
            }

            factor = computed;
            factorPenalty = penalty;
        }

        target.CopyTo(constraint);
        QuadraticChecks.Multiply(d.T, constraint, point);
        for (int i = 0; i < point.Length; i++)
        {
            point[i] = penalty * point[i] - q[i];
        }

        factor!.Solve(point, product);
        new ReadOnlyVectorSpan<double>(product).CopyTo(x);
    }
}

/// <summary>
/// Minimizes the sum of convex functions <c>f(x) + g(z)</c> subject to <c>Dx = z</c> using ADMM.
/// <para>ADMM を用いて制約 <c>Dx = z</c> の下で凸関数の和 <c>f(x) + g(z)</c> を最小化します。</para>
/// </summary>
/// <remarks>
/// <para>In consensus form, <c>D = I</c> and both terms are supplied as proximal operators. Convergence is reached when both the primal residual <c>Dx − z</c> and dual residual <c>ρDᵀ(z − z_old)</c> fall within tolerance. Instances are not thread-safe and must not be called concurrently.</para>
/// <para>合意形式では <c>D = I</c> となり、両項を近接写像として指定します。主残差 <c>Dx − z</c> と双対残差 <c>ρDᵀ(z − z_old)</c> の両方が許容誤差以下になったときに収束します。インスタンスはスレッドセーフではないため、複数スレッドから同時に呼び出さないでください。</para>
/// </remarks>
public sealed class Admm
{
    private readonly double initialPenalty;
    private readonly double absoluteTolerance;
    private readonly double relativeTolerance;
    private readonly double relaxation;
    private readonly bool adaptive;
    private readonly double balance;
    private readonly double factor;
    private readonly int maxPenaltyUpdates;
    private readonly int maxIterations;

    /// <summary>
    /// Initializes a new ADMM solver with the specified options.
    /// <para>指定した設定で ADMM ソルバーを初期化します。</para>
    /// </summary>
    /// <param name="options">Optional solver configuration, or <see langword="null"/> for defaults. / オプション設定。<see langword="null"/> の場合は既定値を使用します。</param>
    /// <exception cref="ArgumentOutOfRangeException">An option value in <paramref name="options"/> is out of range. / <paramref name="options"/> の設定値が有効範囲外の場合。</exception>
    public Admm(AdmmOptions? options = null)
    {
        options ??= new AdmmOptions();
        if (!(options.Penalty > 0 && double.IsFinite(options.Penalty)))
        {
            throw new ArgumentOutOfRangeException(nameof(options.Penalty), "The penalty must be positive and finite.");
        }

        if (!(options.AbsoluteTolerance >= 0))
        {
            throw new ArgumentOutOfRangeException(nameof(options.AbsoluteTolerance), "The tolerance must be non-negative.");
        }

        if (!(options.RelativeTolerance >= 0))
        {
            throw new ArgumentOutOfRangeException(nameof(options.RelativeTolerance), "The tolerance must be non-negative.");
        }

        if (!(options.OverRelaxation > 0 && options.OverRelaxation < 2))
        {
            throw new ArgumentOutOfRangeException(nameof(options.OverRelaxation), "The relaxation must lie in (0, 2).");
        }

        if (!(options.PenaltyBalance > 1))
        {
            throw new ArgumentOutOfRangeException(nameof(options.PenaltyBalance), "The balance must be greater than 1.");
        }

        if (!(options.PenaltyFactor > 1 && double.IsFinite(options.PenaltyFactor)))
        {
            throw new ArgumentOutOfRangeException(nameof(options.PenaltyFactor), "The factor must be greater than 1 and finite.");
        }

        ArgumentOutOfRangeException.ThrowIfNegative(options.MaxPenaltyUpdates, nameof(options.MaxPenaltyUpdates));
        ArgumentOutOfRangeException.ThrowIfLessThan(options.MaxIterations, 1, nameof(options.MaxIterations));
        initialPenalty = options.Penalty;
        absoluteTolerance = options.AbsoluteTolerance;
        relativeTolerance = options.RelativeTolerance;
        relaxation = options.OverRelaxation;
        adaptive = options.AdaptivePenalty;
        balance = options.PenaltyBalance;
        factor = options.PenaltyFactor;
        maxPenaltyUpdates = options.MaxPenaltyUpdates;
        maxIterations = options.MaxIterations;
    }

    /// <summary>
    /// Solves the consensus-form problem <c>min f(x) + g(z)</c> subject to <c>x = z</c>.
    /// <para>合意形式の最適化問題 <c>min f(x) + g(z)</c>（<c>x = z</c>）を解きます。</para>
    /// </summary>
    /// <typeparam name="TF">The proximal operator type for <c>f</c>. / 第1項 <c>f</c> の近接写像の型。</typeparam>
    /// <typeparam name="TG">The proximal operator type for <c>g</c>. / 第2項 <c>g</c> の近接写像の型。</typeparam>
    /// <param name="f">The first proximable convex function <c>f</c>. / 近接写像を持つ1つ目の凸関数 <c>f</c>。</param>
    /// <param name="g">The second proximable convex function <c>g</c>. / 近接写像を持つ2つ目の凸関数 <c>g</c>。</param>
    /// <param name="initialPoint">The finite non-empty initial value for <c>x</c> and <c>z</c> (<c>u</c> starts at 0). / <c>x</c> と <c>z</c> の初期値（長さ 1 以上の有限値。<c>u</c> の初期値は 0）。</param>
    /// <param name="cancellationToken">A cancellation token checked at each iteration. / 反復ごとに確認されるキャンセルトークン。</param>
    /// <returns>An <see cref="AdmmResult"/> containing the primal, split, and scaled dual variables, objective value, residuals, and termination reason. / 主変数、分割変数、スケール化双対変数、目的関数値、残差、および終了理由を含む <see cref="AdmmResult"/>。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="f"/> or <paramref name="g"/> is <see langword="null"/>. / <paramref name="f"/> または <paramref name="g"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoint"/> is empty or contains a non-finite value. / <paramref name="initialPoint"/> が空、または有限でない値を含む場合。</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was canceled. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public AdmmResult Minimize<TF, TG>(TF f, TG g, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)
        where TF : IProximalOperator
        where TG : IProximalOperator
    {
        ArgumentNullException.ThrowIfNull(f);
        ArgumentNullException.ThrowIfNull(g);
        CheckPoint(initialPoint, initialPoint.Length);
        return Run(new ConsensusSubproblem<TF>(f, initialPoint.Length), g, initialPoint, cancellationToken);
    }

    /// <summary>
    /// Solves the splitting-form problem <c>min f(x) + g(z)</c> subject to <c>Dx = z</c>.
    /// <para>分割形式の最適化問題 <c>min f(x) + g(z)</c>（<c>Dx = z</c>）を解きます。</para>
    /// </summary>
    /// <typeparam name="TG">The proximal operator type for <c>g</c>. / 第2項 <c>g</c> の近接写像の型。</typeparam>
    /// <param name="f">The subproblem handling the <c>x</c>-update and linear operator <c>D</c>. / <c>x</c> 更新と線形写像 <c>D</c> を担当する部分問題。</param>
    /// <param name="g">The proximable convex function <c>g</c> acting on vectors of length <c>p</c>. / 長さ <c>p</c> のベクトルに作用する近接写像付き凸関数 <c>g</c>。</param>
    /// <param name="initialPoint">The finite initial value for <c>x</c> of length <c>n</c> (<c>z</c> starts at <c>Dx</c> and <c>u</c> starts at 0). / 長さ <c>n</c> の有限の初期値 <c>x</c>（<c>z</c> は <c>Dx</c>、<c>u</c> は 0 で初期化されます）。</param>
    /// <param name="cancellationToken">A cancellation token checked at each iteration. / 反復ごとに確認されるキャンセルトークン。</param>
    /// <returns>An <see cref="AdmmResult"/> containing the primal, split, and scaled dual variables, objective value, residuals, and termination reason. / 主変数、分割変数、スケール化双対変数、目的関数値、残差、および終了理由を含む <see cref="AdmmResult"/>。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="f"/> or <paramref name="g"/> is <see langword="null"/>. / <paramref name="f"/> または <paramref name="g"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoint"/> does not have length <c>n</c> or contains a non-finite value. / <paramref name="initialPoint"/> の長さが <c>n</c> と一致しない、または有限でない値を含む場合。</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was canceled. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public AdmmResult Minimize<TG>(IAdmmSubproblem f, TG g, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)
        where TG : IProximalOperator
    {
        ArgumentNullException.ThrowIfNull(f);
        ArgumentNullException.ThrowIfNull(g);
        CheckPoint(initialPoint, f.Dimension);
        return Run(f, g, initialPoint, cancellationToken);
    }

    private static void CheckPoint(ReadOnlySpan<double> initialPoint, int dimension)
    {
        if (initialPoint.IsEmpty || initialPoint.Length != dimension)
        {
            throw new ArgumentException($"The initial point must have length {Math.Max(dimension, 1)}.", nameof(initialPoint));
        }

        foreach (double v in initialPoint)
        {
            if (!double.IsFinite(v))
            {
                throw new ArgumentException("The initial point must be finite.", nameof(initialPoint));
            }
        }
    }

    private AdmmResult Run<TSub, TG>(TSub f, TG g, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken)
        where TSub : IAdmmSubproblem
        where TG : IProximalOperator
    {
        int n = f.Dimension, p = f.ConstraintCount;
        var x = initialPoint.ToArray();
        var dx = new double[p];
        var z = new double[p];
        var zOld = new double[p];
        var u = new double[p];
        var work = new double[p];
        var dual = new double[n];
        f.Multiply(x, dx);
        dx.CopyTo(z, 0);

        double rho = initialPenalty;
        double primal = double.NaN, dualResidual = double.NaN;
        int updates = 0, iterations = 0;
        var termination = MinimizationTermination.MaxIterations;
        double sqrtP = Math.Sqrt(p), sqrtN = Math.Sqrt(n);
        while (iterations < maxIterations)
        {
            cancellationToken.ThrowIfCancellationRequested();
            iterations++;

            // x の更新
            for (int i = 0; i < p; i++)
            {
                work[i] = z[i] - u[i];
            }

            f.Solve(work, rho, x);
            f.Multiply(x, dx);

            // z の更新（過緩和した Dx を使う）
            z.CopyTo(zOld, 0);
            for (int i = 0; i < p; i++)
            {
                work[i] = relaxation * dx[i] + (1 - relaxation) * zOld[i] + u[i];
            }

            g.Apply(work, 1 / rho, z);

            // u の更新と残差
            double r2 = 0, dxNorm2 = 0, zNorm2 = 0;
            bool finite = true;
            for (int i = 0; i < p; i++)
            {
                double relaxed = relaxation * dx[i] + (1 - relaxation) * zOld[i];
                u[i] += relaxed - z[i];
                double r = dx[i] - z[i];
                r2 += r * r;
                dxNorm2 += dx[i] * dx[i];
                zNorm2 += z[i] * z[i];
                work[i] = z[i] - zOld[i];
                finite &= double.IsFinite(z[i]) && double.IsFinite(u[i]);
            }

            if (!finite || !double.IsFinite(r2))
            {
                termination = MinimizationTermination.NonFiniteValue;
                break;
            }

            f.MultiplyTransposed(work, dual);
            double s2 = 0;
            foreach (double v in dual)
            {
                s2 += v * v;
            }

            f.MultiplyTransposed(u, dual);
            double yNorm2 = 0;
            foreach (double v in dual)
            {
                yNorm2 += v * v;
            }

            primal = Math.Sqrt(r2);
            dualResidual = rho * Math.Sqrt(s2);
            double primalTolerance = sqrtP * absoluteTolerance + relativeTolerance * Math.Sqrt(Math.Max(dxNorm2, zNorm2));
            double dualTolerance = sqrtN * absoluteTolerance + relativeTolerance * rho * Math.Sqrt(yNorm2);
            if (primal <= primalTolerance && dualResidual <= dualTolerance)
            {
                termination = MinimizationTermination.ResidualTolerance;
                break;
            }

            if (adaptive && updates < maxPenaltyUpdates)
            {
                double scale = 1;
                if (primal > balance * dualResidual)
                {
                    scale = factor;
                }
                else if (dualResidual > balance * primal)
                {
                    scale = 1 / factor;
                }

                if (scale != 1)
                {
                    rho *= scale;
                    for (int i = 0; i < p; i++)
                    {
                        u[i] /= scale;
                    }

                    updates++;
                }
            }
        }

        double value = f.Evaluate(x) + g.Evaluate(z);
        return new AdmmResult(new Vector<double>(x), new Vector<double>(z), new Vector<double>(u), value, primal, dualResidual, rho, iterations, termination);
    }

    // 合意形式: D = I、x の更新は f の近接写像
    private readonly struct ConsensusSubproblem<TF>(TF proximal, int dimension) : IAdmmSubproblem
        where TF : IProximalOperator
    {
        public int Dimension => dimension;

        public int ConstraintCount => dimension;

        public double Evaluate(ReadOnlyVectorSpan<double> x) => proximal.Evaluate(x);

        public void Multiply(ReadOnlyVectorSpan<double> x, VectorSpan<double> result) => x.CopyTo(result);

        public void MultiplyTransposed(ReadOnlyVectorSpan<double> y, VectorSpan<double> result) => y.CopyTo(result);

        public void Solve(ReadOnlyVectorSpan<double> target, double penalty, VectorSpan<double> x) => proximal.Apply(target, 1 / penalty, x);
    }
}
