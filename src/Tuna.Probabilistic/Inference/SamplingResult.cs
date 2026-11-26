using Tuna.LinearAlgebra;

namespace Tuna.Probabilistic.Inference;

/// <summary>
/// Represents the post-warmup draws and evaluation statistics produced by an MCMC sampler.
/// <para>MCMC サンプラーによって生成されたウォームアップ後の標本と評価統計量を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// <see cref="Draws"/> is a 3D tensor with shape <c>[dimension, draw, chain]</c> following the library-wide column-per-point convention.
/// <see cref="Chain(int)"/> returns the <c>dimension × draw</c> matrix for a single chain, and <see cref="Parameter(int)"/> returns the <c>draw × chain</c> matrix for a single parameter (the input shape expected by <see cref="ConvergenceDiagnostics"/>).
/// Sampler-specific diagnostics (acceptance rates, tree depths, temperature ladders, etc.) are exposed by derived types.
/// </para>
/// <para>
/// <see cref="Draws"/> は <c>[次元, 反復, チェーン]</c> 形状の 3 階テンソルであり、ライブラリ共通の「1 列 = 1 点」の形式に従います。
/// <see cref="Chain(int)"/> は単一チェーンの <c>次元 × 反復</c> 行列を、<see cref="Parameter(int)"/> は単一パラメータの <c>反復 × チェーン</c> 行列（<see cref="ConvergenceDiagnostics"/> の入力形式）を返します。
/// サンプラー固有の診断情報（受理率、木の深さ、温度ラダーなど）は派生クラスで公開されます。
/// </para>
/// </remarks>
public class SamplingResult
{
    private readonly long[] evaluations;

    internal SamplingResult(Tensor<double> draws, Matrix<double> logDensities, long[] evaluations, ulong seed)
    {
        Draws = draws;
        LogDensities = logDensities;
        this.evaluations = evaluations;
        Seed = seed;
    }

    /// <summary>
    /// Gets the post-warmup draws tensor with shape <c>[dimension, draw, chain]</c>.
    /// <para>ウォームアップ後に保持された標本テンソル（形状 <c>[次元, 反復, チェーン]</c>）を取得します。</para>
    /// </summary>
    /// <value>A tensor where element <c>(i, t, c)</c> is component <c>i</c> of draw <c>t</c> in chain <c>c</c>. / 要素 <c>(i, t, c)</c> が第 <c>c</c> チェーンの第 <c>t</c> 標本の第 <c>i</c> 成分であるテンソル。</value>
    public Tensor<double> Draws { get; }

    /// <summary>
    /// Gets the log-density evaluated at each retained draw, with shape <c>[draw × chain]</c>.
    /// <para>保持された各標本における対数密度行列（形状 <c>[反復 × チェーン]</c>）を取得します。</para>
    /// </summary>
    /// <value>The log-density values as evaluated by the sampler, including the log-Jacobian term when a bijector transform is used. / サンプラーが評価した対数密度の行列。制約変換を使用している場合はヤコビアン項を含みます。</value>
    public Matrix<double> LogDensities { get; }

    /// <summary>
    /// Gets the parameter dimension.
    /// <para>パラメータの次元を取得します。</para>
    /// </summary>
    public int Dimension => (int)Draws.Lengths[0];

    /// <summary>
    /// Gets the number of retained draws per chain.
    /// <para>チェーンごとの保持標本数を取得します。</para>
    /// </summary>
    public int DrawCount => (int)Draws.Lengths[1];

    /// <summary>
    /// Gets the number of chains (or walkers in <see cref="EnsembleSampler"/>).
    /// <para>チェーン数（<see cref="EnsembleSampler"/> ではウォーカー数）を取得します。</para>
    /// </summary>
    public int ChainCount => (int)Draws.Lengths[2];

    /// <summary>
    /// Gets the random seed used for sampling (either configured or obtained from the OS).
    /// <para>サンプリングに使用した乱数シード（指定値、または省略時に OS から取得した値）を取得します。</para>
    /// </summary>
    public ulong Seed { get; }

    /// <summary>
    /// Gets the total number of log-density evaluations (including gradient evaluations and warmup) for each chain.
    /// <para>チェーンごとの対数密度評価回数（勾配評価およびウォームアップ期間を含む）を取得します。</para>
    /// </summary>
    public IReadOnlyList<long> Evaluations => evaluations;

    /// <summary>
    /// Gets the total number of log-density evaluations summed across all chains.
    /// <para>全チェーン合計の対数密度評価回数を取得します。</para>
    /// </summary>
    public long TotalEvaluations
    {
        get
        {
            long total = 0;
            foreach (long e in evaluations)
            {
                total += e;
            }

            return total;
        }
    }

    /// <summary>
    /// Returns a read-only view of all draws for the specified parameter.
    /// <para>指定したパラメータの標本に対する読み取り専用ビューを取得します。</para>
    /// </summary>
    /// <param name="index">The zero-based parameter index. / 0 から始まるパラメータのインデックス。</param>
    /// <returns>A read-only matrix view of shape <c>[draw × chain]</c> sharing memory with <see cref="Draws"/>. / <see cref="Draws"/> とメモリを共有する <c>[反復 × チェーン]</c> の読み取り専用行列ビュー。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is out of range. / <paramref name="index"/> が範囲外の場合。</exception>
    public ReadOnlyMatrixSpan<double> Parameter(int index)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, Dimension);
        return ParameterView(Draws, index);
    }

    /// <summary>
    /// Returns a matrix view of all draws for the specified chain.
    /// <para>指定したチェーンの標本に対する行列ビューを取得します。</para>
    /// </summary>
    /// <param name="chain">The zero-based chain index. / 0 から始まるチェーンのインデックス。</param>
    /// <returns>A matrix view of shape <c>[dimension × draw]</c> sharing memory with <see cref="Draws"/>, where each column is one draw. / <see cref="Draws"/> とメモリを共有する <c>[次元 × 反復]</c> の行列ビュー（各列が 1 標本）。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="chain"/> is out of range. / <paramref name="chain"/> が範囲外の場合。</exception>
    public MatrixSpan<double> Chain(int chain)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(chain);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(chain, ChainCount);
        return Draws.GetMatrix(chain);
    }

    /// <summary>
    /// Computes the sample mean pooled across all chains.
    /// <para>全チェーンを統合した標本の平均ベクトルを計算します。</para>
    /// </summary>
    /// <returns>A new vector containing the estimated posterior mean of each parameter. / 各パラメータの事後平均推定値を格納した新しいベクトル。</returns>
    public Vector<double> Mean()
    {
        int d = Dimension;
        var mean = new Vector<double>(d);
        Span<double> data = Draws.AsSpan();
        long count = (long)DrawCount * ChainCount;
        for (long k = 0; k < count; k++)
        {
            Span<double> point = data.Slice((int)(k * d), d);
            for (int i = 0; i < d; i++)
            {
                mean[i] += point[i];
            }
        }

        for (int i = 0; i < d; i++)
        {
            mean[i] /= count;
        }

        return mean;
    }

    /// <summary>
    /// Computes the unbiased sample covariance matrix pooled across all chains.
    /// <para>全チェーンを統合した標本の不偏共分散行列を計算します。</para>
    /// </summary>
    /// <returns>The sample covariance matrix divided by <c>N − 1</c>, or a matrix of <see cref="double.NaN"/> when only one draw exists. / 標本数 <c>N − 1</c> で割った不偏共分散行列。総標本数が 1 の場合は全要素が <see cref="double.NaN"/>。</returns>
    public Matrix<double> Covariance()
    {
        int d = Dimension;
        Vector<double> mean = Mean();
        var covariance = new Matrix<double>(d, d);
        Span<double> data = Draws.AsSpan();
        long count = (long)DrawCount * ChainCount;
        var centered = new double[d];
        for (long k = 0; k < count; k++)
        {
            Span<double> point = data.Slice((int)(k * d), d);
            for (int i = 0; i < d; i++)
            {
                centered[i] = point[i] - mean[i];
            }

            for (int j = 0; j < d; j++)
            {
                for (int i = j; i < d; i++)
                {
                    covariance[i, j] += centered[i] * centered[j];
                }
            }
        }

        for (int j = 0; j < d; j++)
        {
            for (int i = j; i < d; i++)
            {
                double value = covariance[i, j] / (count - 1);
                covariance[i, j] = value;
                covariance[j, i] = value;
            }
        }

        return covariance;
    }

    /// <summary>
    /// Computes per-parameter posterior summary statistics and convergence diagnostics.
    /// <para>標本からパラメータごとの事後分布の要約統計量と収束診断を計算します。</para>
    /// </summary>
    /// <param name="names">Optional parameter names; defaults to <c>theta[0]</c>, <c>theta[1]</c>, etc. / 省略可能なパラメータ名一覧（省略時は <c>theta[0]</c>, <c>theta[1]</c>, ...）。</param>
    /// <returns>A <see cref="PosteriorSummary"/> table for all parameters. / 全パラメータの要約と収束診断を格納した <see cref="PosteriorSummary"/>。</returns>
    /// <exception cref="ArgumentException">The number of names does not match <see cref="Dimension"/>. / パラメータ名の数が <see cref="Dimension"/> と一致しない場合。</exception>
    public PosteriorSummary Summarize(IReadOnlyList<string>? names = null) => PosteriorSummary.Compute(Draws, names);

    /// <summary>
    /// Maps all draws into the constrained parameter space using the specified bijector.
    /// <para>指定した制約変換を用いて、すべての標本を制約付きパラメータ空間へ写像します。</para>
    /// </summary>
    /// <typeparam name="TBijector">The bijector type. / 制約変換の型。</typeparam>
    /// <param name="bijector">The bijector mapping unconstrained draws to constrained space. / 制約なし空間の標本を制約付き空間へ写像する変換。</param>
    /// <returns>A new <see cref="SamplingResult"/> containing the transformed draws along with cloned log-densities, evaluation counts, and seed. Derived-class diagnostics are not preserved. / 変換後の標本と、元の対数密度・評価回数・シードのコピーを含む新しい <see cref="SamplingResult"/>。派生クラス固有の診断情報は含まれません。</returns>
    /// <remarks>
    /// <para><see cref="LogDensities"/> is copied unchanged and therefore still represents the unconstrained log-density including the log-Jacobian term.</para>
    /// <para><see cref="LogDensities"/> は変換前の値（ヤコビアン項を含む制約なし空間の対数密度）がそのままコピーされます。</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="bijector"/> is <see langword="null"/>. / <paramref name="bijector"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><see cref="IBijector.UnconstrainedDimension"/> does not equal <see cref="Dimension"/>. / <see cref="IBijector.UnconstrainedDimension"/> が <see cref="Dimension"/> と一致しない場合。</exception>
    public SamplingResult Transform<TBijector>(TBijector bijector) where TBijector : IBijector
    {
        ArgumentNullException.ThrowIfNull(bijector);
        if (bijector.UnconstrainedDimension != Dimension)
        {
            throw new ArgumentException($"The bijector must have unconstrained dimension {Dimension}.", nameof(bijector));
        }

        int d = Dimension, k = bijector.ConstrainedDimension;
        var constrained = new Tensor<double>(k, DrawCount, ChainCount);
        Span<double> source = Draws.AsSpan(), destination = constrained.AsSpan();
        long count = (long)DrawCount * ChainCount;
        for (long p = 0; p < count; p++)
        {
            bijector.Forward(source.Slice((int)(p * d), d), destination.Slice((int)(p * k), k));
        }

        return new SamplingResult(constrained, LogDensities.Clone(), (long[])evaluations.Clone(), Seed);
    }

    internal static ReadOnlyMatrixSpan<double> ParameterView(Tensor<double> draws, int index)
    {
        nint d = draws.Lengths[0], n = draws.Lengths[1], m = draws.Lengths[2];
        return new ReadOnlyMatrixSpan<double>(draws.AsSpan()[index..], new nint[] { n, m }, new nint[] { d, d * n });
    }
}
