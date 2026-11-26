using Tuna.Integration.Internal;
using Tuna.LinearAlgebra;
using Tuna.QuasiRandom;
using Tuna.RandomSource;

namespace Tuna.Integration;

/// <summary>
/// Specifies options for randomized quasi-Monte Carlo integration (<see cref="QuasiMonteCarlo"/>).
/// <para>乱択化準モンテカルロ積分（<see cref="QuasiMonteCarlo"/>）の設定を表します。</para>
/// </summary>
/// <remarks>
/// <para>The number of points per replicate doubles from <see cref="InitialPoints"/> until the standard error is at most <c>max(AbsoluteTolerance, RelativeTolerance * |value|)</c> or the next doubling would exceed <see cref="MaxEvaluations"/>. Set <see cref="MaxEvaluations"/> to <c>Replicates * InitialPoints</c> to stop after the initial sample size.</para>
/// <para>各複製の点数を <see cref="InitialPoints"/> から2倍ずつ増やし、標準誤差が <c>max(AbsoluteTolerance, RelativeTolerance * |value|)</c> 以下になるか、次の段階の評価回数が <see cref="MaxEvaluations"/> を超えると終了します。初期点数のみで打ち切るには <see cref="MaxEvaluations"/> に <c>Replicates * InitialPoints</c> を設定してください。</para>
/// </remarks>
public sealed class QuasiMonteCarloOptions
{
    /// <summary>
    /// Gets the number of independently randomized Sobol' sequences.
    /// <para>独立に乱択化する Sobol' 列の複製数を取得します。</para>
    /// </summary>
    /// <value>An integer greater than or equal to 2. The default is 16. / 2 以上の整数。既定値は 16 です。</value>
    public int Replicates { get; init; } = 16;

    /// <summary>
    /// Gets the initial number of points evaluated per replicate.
    /// <para>各複製で最初に評価する点数を取得します。</para>
    /// </summary>
    /// <value>A positive power of two. The default is 1,024. / 正の 2 のべき乗。既定値は 1,024 です。</value>
    public int InitialPoints { get; init; } = 1024;

    /// <summary>
    /// Gets the absolute tolerance for the standard error.
    /// <para>標準誤差の絶対許容誤差を取得します。</para>
    /// </summary>
    /// <value>A finite, non-negative value. The default is 0. / 0 以上の有限値。既定値は 0 です。</value>
    public double AbsoluteTolerance { get; init; }

    /// <summary>
    /// Gets the relative tolerance for the standard error.
    /// <para>標準誤差の相対許容誤差を取得します。</para>
    /// </summary>
    /// <value>A finite, non-negative value. The default is 1e-4. / 0 以上の有限値。既定値は 1e-4 です。</value>
    public double RelativeTolerance { get; init; } = 1e-4;

    /// <summary>
    /// Gets the maximum total number of integrand evaluations.
    /// <para>被積分関数の総評価回数の上限を取得します。</para>
    /// </summary>
    /// <value>A value greater than or equal to <c>Replicates * InitialPoints</c>. The default is <c>2^24</c>. / <c>Replicates * InitialPoints</c> 以上の値。既定値は <c>2^24</c> です。</value>
    public long MaxEvaluations { get; init; } = 1L << 24;

    internal static readonly QuasiMonteCarloOptions Default = new();

    internal void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(Replicates, 2, nameof(Replicates));
        if (InitialPoints < 1 || (InitialPoints & (InitialPoints - 1)) != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(InitialPoints), "The initial number of points must be a power of two.");
        }

        if (!(AbsoluteTolerance >= 0 && double.IsFinite(AbsoluteTolerance)))
        {
            throw new ArgumentOutOfRangeException(nameof(AbsoluteTolerance), "The absolute tolerance must be finite and non-negative.");
        }

        if (!(RelativeTolerance >= 0 && double.IsFinite(RelativeTolerance)))
        {
            throw new ArgumentOutOfRangeException(nameof(RelativeTolerance), "The relative tolerance must be finite and non-negative.");
        }

        if (MaxEvaluations < (long)Replicates * InitialPoints)
        {
            throw new ArgumentOutOfRangeException(nameof(MaxEvaluations), "The evaluation limit must allow the initial points of every replicate.");
        }
    }
}

/// <summary>
/// Represents the result of randomized quasi-Monte Carlo integration.
/// <para>乱択化準モンテカルロ積分の結果を表します。</para>
/// </summary>
/// <param name="Value">The estimated integral value (mean of replicate estimates). / 積分の推定値（各複製の推定値の平均）。</param>
/// <param name="StandardError">The standard error of the estimate (sample standard deviation across replicates divided by the square root of <paramref name="Replicates"/>). / 推定値の標準誤差（複製間の標本標準偏差を <paramref name="Replicates"/> の平方根で割った値）。</param>
/// <param name="Replicates">The number of independent replicates. / 独立な複製数。</param>
/// <param name="PointsPerReplicate">The number of points used per replicate (or the target stage count when <paramref name="Termination"/> is <see cref="IntegrationTermination.NonFiniteValue"/>). / 各複製で使用した点数（<paramref name="Termination"/> が <see cref="IntegrationTermination.NonFiniteValue"/> の場合は評価していた段階の点数）。</param>
/// <param name="Evaluations">The total number of integrand evaluations performed, typically <c>Replicates * PointsPerReplicate</c>. / 被積分関数を評価した総回数（通常は <c>Replicates * PointsPerReplicate</c>）。</param>
/// <param name="Termination">The reason integration terminated (<see cref="IntegrationTermination.Tolerance"/>, <see cref="IntegrationTermination.MaxEvaluations"/>, or <see cref="IntegrationTermination.NonFiniteValue"/>). / 終了理由（<see cref="IntegrationTermination.Tolerance"/>、<see cref="IntegrationTermination.MaxEvaluations"/>、または <see cref="IntegrationTermination.NonFiniteValue"/>）。</param>
public readonly record struct QuasiMonteCarloResult(
    double Value, double StandardError, int Replicates, long PointsPerReplicate, long Evaluations, IntegrationTermination Termination)
{
    /// <summary>
    /// Gets a value indicating whether integration converged within the requested tolerance.
    /// <para>積分が許容誤差内に収束したかどうかを取得します。</para>
    /// </summary>
    /// <value><see langword="true"/> if <see cref="Termination"/> is <see cref="IntegrationTermination.Tolerance"/>; otherwise, <see langword="false"/>. / <see cref="Termination"/> が <see cref="IntegrationTermination.Tolerance"/> の場合は <see langword="true"/>、それ以外は <see langword="false"/>。</value>
    public bool Converged => Termination == IntegrationTermination.Tolerance;
}

/// <summary>
/// Provides randomized quasi-Monte Carlo integration over the unit hypercube <c>(0, 1)^d</c>.
/// <para>単位超立方体 <c>(0, 1)^d</c> 上の乱択化準モンテカルロ積分を提供します。</para>
/// </summary>
/// <remarks>
/// <para>Sample coordinates lie in <c>[2^-53, 1 - 2^-53]</c> and never include 0 or 1. To integrate over another domain, apply a change of variables and include the Jacobian in the integrand. Assign the most influential variables to earlier dimensions.</para>
/// <para>Results are reproducible given the same <see cref="IRandomSource"/> state and integrand. The integrand is evaluated on the calling thread, and any non-finite evaluation or sum terminates with <see cref="IntegrationTermination.NonFiniteValue"/>.</para>
/// <para>標本点の座標は <c>[2^-53, 1 - 2^-53]</c> の範囲にあり、0 と 1 は含まれません。単位超立方体以外の領域を積分する場合は変数変換を行い、ヤコビアンを被積分関数に含めてください。寄与の大きい変数ほど前の次元に割り当てます。</para>
/// <para>同じ状態の <see cref="IRandomSource"/> と同じ被積分関数を用いれば結果を再現できます。被積分関数は呼び出し元スレッドで評価され、評価値または和が有限値でなくなると <see cref="IntegrationTermination.NonFiniteValue"/> で終了します。</para>
/// </remarks>
public static class QuasiMonteCarlo
{
    // 1回に生成する点数（d × Block 行列）
    private const int Block = 256;

    /// <summary>
    /// Estimates the integral <c>∫_(0,1)^d f(u) du</c> over the unit hypercube.
    /// <para>単位超立方体上の積分 <c>∫_(0,1)^d f(u) du</c> を推定します。</para>
    /// </summary>
    /// <param name="integrand">The integrand function. / 被積分関数。</param>
    /// <param name="dimension">The number of dimensions, from 1 to <see cref="SobolSequence.MaxDimension"/>. / 次元数（1 以上 <see cref="SobolSequence.MaxDimension"/> 以下）。</param>
    /// <param name="random">The random source used for scrambling. / 乱択化に使用する乱数源。</param>
    /// <param name="options">Integration options, or <see langword="null"/> to use default settings. / 積分の設定。<see langword="null"/> の場合は既定値を使用します。</param>
    /// <returns>The estimated integral, standard error, sample counts, and termination reason. / 積分の推定値、標準誤差、評価点数、および終了理由。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="integrand"/> or <paramref name="random"/> is <see langword="null"/>. / <paramref name="integrand"/> または <paramref name="random"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="dimension"/> is out of range, or an option value in <paramref name="options"/> is invalid. / <paramref name="dimension"/> が範囲外の場合、または <paramref name="options"/> の設定値が範囲外の場合。</exception>
    public static QuasiMonteCarloResult Integrate(Func<ReadOnlyVectorSpan<double>, double> integrand, int dimension, IRandomSource random,
        QuasiMonteCarloOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(integrand);
        return Integrate(new DelegateMultivariateIntegrand(integrand), dimension, random, options);
    }

    /// <summary>
    /// Estimates the integral <c>∫_(0,1)^d f(u) du</c> over the unit hypercube.
    /// <para>単位超立方体上の積分 <c>∫_(0,1)^d f(u) du</c> を推定します。</para>
    /// </summary>
    /// <typeparam name="TIntegrand">The type of the integrand. / 被積分関数の型。</typeparam>
    /// <inheritdoc cref="Integrate(Func{ReadOnlyVectorSpan{double}, double}, int, IRandomSource, QuasiMonteCarloOptions?)"/>
    public static QuasiMonteCarloResult Integrate<TIntegrand>(TIntegrand integrand, int dimension, IRandomSource random,
        QuasiMonteCarloOptions? options = null)
        where TIntegrand : IMultivariateIntegrand
    {
        ArgumentNullException.ThrowIfNull(integrand);
        ArgumentNullException.ThrowIfNull(random);
        ArgumentOutOfRangeException.ThrowIfLessThan(dimension, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(dimension, SobolSequence.MaxDimension);
        options ??= QuasiMonteCarloOptions.Default;
        options.Validate();

        int replicates = options.Replicates;
        var sequences = new SobolSequence[replicates];
        for (int r = 0; r < replicates; r++)
        {
            sequences[r] = new SobolSequence(dimension, random);
        }

        var sums = new CompensatedSum[replicates];
        var points = new Matrix<double>(dimension, Math.Min(Block, options.InitialPoints));
        long used = 0;
        long next = options.InitialPoints;
        long evaluations = 0;
        while (true)
        {
            // 各複製について [used, next) の標本点を評価して累積する
            for (int r = 0; r < replicates; r++)
            {
                for (long start = used; start < next; start += points.Columns)
                {
                    var block = sequences[r].GetPoints((ulong)start, points.AsMatrixSpan());
                    for (int j = 0; j < block.Columns; j++)
                    {
                        double v = integrand.Evaluate(block.Column(j));
                        evaluations++;
                        if (!double.IsFinite(v))
                        {
                            return new QuasiMonteCarloResult(double.NaN, double.NaN, replicates, next, evaluations, IntegrationTermination.NonFiniteValue);
                        }

                        sums[r].Add(v);
                    }
                }
            }

            used = next;
            var (value, standardError) = Estimate(sums, used);
            if (!double.IsFinite(value) || !double.IsFinite(standardError))
            {
                // 累積和が double の範囲を超過した
                return new QuasiMonteCarloResult(double.NaN, double.NaN, replicates, used, evaluations, IntegrationTermination.NonFiniteValue);
            }

            double target = Math.Max(options.AbsoluteTolerance, options.RelativeTolerance * Math.Abs(value));
            if (standardError <= target)
            {
                return new QuasiMonteCarloResult(value, standardError, replicates, used, evaluations, IntegrationTermination.Tolerance);
            }

            if (used > options.MaxEvaluations / (2L * replicates))
            {
                return new QuasiMonteCarloResult(value, standardError, replicates, used, evaluations, IntegrationTermination.MaxEvaluations);
            }

            next = 2 * used;
            if (points.Columns < Block)
            {
                points = new Matrix<double>(dimension, (int)Math.Min(Block, used));
            }
        }
    }

    private static (double Value, double StandardError) Estimate(CompensatedSum[] sums, long points)
    {
        int r = sums.Length;
        Span<double> means = r <= 64 ? stackalloc double[r] : new double[r];
        var total = new CompensatedSum();
        for (int i = 0; i < r; i++)
        {
            means[i] = sums[i].Value / points;
            total.Add(means[i]);
        }

        double mean = total.Value / r;

        // 偏差の二乗でのオーバーフロー・アンダーフローを防ぐため、最大偏差で正規化してから二乗和を計算する
        double largest = 0;
        for (int i = 0; i < r; i++)
        {
            largest = Math.Max(largest, Math.Abs(means[i] - mean));
        }

        if (largest == 0)
        {
            return (mean, 0);
        }

        var squares = new CompensatedSum();
        for (int i = 0; i < r; i++)
        {
            double deviation = (means[i] - mean) / largest;
            squares.Add(deviation * deviation);
        }

        return (mean, largest * Math.Sqrt(squares.Value / ((double)r * (r - 1))));
    }
}
