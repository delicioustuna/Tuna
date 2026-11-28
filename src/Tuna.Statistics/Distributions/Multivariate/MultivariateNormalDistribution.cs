using Tuna.Integration;
using Tuna.LinearAlgebra;
using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the multivariate normal (Gaussian) distribution <c>N(μ, Σ)</c>.
/// <para>多変量正規分布（ガウス分布）<c>N(μ, Σ)</c> を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// The covariance matrix must be finite, symmetric within rounding tolerance (<c>|σᵢⱼ - σⱼᵢ| ≤ √ε √σᵢᵢ √σⱼⱼ</c>, lower triangle used), and numerically positive definite (verified by Cholesky factorization; singular positive semidefinite matrices are rejected).
/// Density evaluation returns 0 for points containing infinities or where <c>L⁻¹(x - μ)</c> overflows.
/// In 1 dimension, a distribution constructed via <see cref="FromCholeskyFactor"/> with the standard deviation produces identical samples to <see cref="NormalDistribution"/>.
/// </para>
/// <para>
/// 共分散行列は有限かつ丸め誤差の範囲内で対称（<c>|σᵢⱼ - σⱼᵢ| ≤ √ε √σᵢᵢ √σⱼⱼ</c>、下三角を使用）であり、コレスキー分解が成功する数値的に正定値な行列である必要があります（特異な半正定値行列は受け付けません）。
/// 評価点に無限大を含む場合や <c>L⁻¹(x - μ)</c> がオーバーフローする遠方では密度 0 を返します。
/// 1 次元で <see cref="FromCholeskyFactor"/> に標準偏差を指定した場合、サンプリング結果は <see cref="NormalDistribution"/> と一致します。
/// </para>
/// </remarks>
public sealed partial class MultivariateNormalDistribution : IMultivariateContinuousDistribution
{
    private readonly double[] _mean;
    private readonly double[] _covariance;
    private readonly double[] _factor;
    private readonly double _halfLogDeterminant;

    /// <summary>
    /// Initializes a new instance of <see cref="MultivariateNormalDistribution"/> with the specified mean vector and covariance matrix.
    /// <para>平均ベクトルと共分散行列を指定して <see cref="MultivariateNormalDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="mean">The finite mean vector <c>μ</c> (length at least 1). / 長さ 1 以上の有限な平均ベクトル <c>μ</c>。</param>
    /// <param name="covariance">The finite <c>d × d</c> symmetric positive definite covariance matrix <c>Σ</c> (lower triangle used). / 有限な <c>d × d</c> 対称正定値共分散行列 <c>Σ</c>（下三角を使用）。</param>
    /// <exception cref="ArgumentException"><paramref name="mean"/> is empty, or <paramref name="covariance"/> has a mismatched shape, is asymmetric, or is not numerically positive definite. / <paramref name="mean"/> が空、または <paramref name="covariance"/> の形状不一致・非対称・非正定値の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="mean"/> or <paramref name="covariance"/> contains non-finite entries. / <paramref name="mean"/> または <paramref name="covariance"/> に非有限の要素が含まれる場合。</exception>
    public MultivariateNormalDistribution(ReadOnlyVectorSpan<double> mean, ReadOnlyMatrixSpan<double> covariance)
    {
        _mean = MultivariateMath.FiniteVector(mean, nameof(mean));
        int d = _mean.Length;
        _covariance = MultivariateMath.SymmetricMatrix(covariance, d, nameof(covariance));
        _factor = MultivariateMath.PositiveDefiniteFactor(_covariance, d, nameof(covariance));
        _halfLogDeterminant = MultivariateMath.SumLogDiagonal(_factor, d);
    }

    private MultivariateNormalDistribution(double[] mean, double[] covariance, double[] factor)
    {
        _mean = mean;
        _covariance = covariance;
        _factor = factor;
        _halfLogDeterminant = MultivariateMath.SumLogDiagonal(factor, mean.Length);
    }

    /// <summary>
    /// Creates a <see cref="MultivariateNormalDistribution"/> from a mean vector and lower-triangular Cholesky factor <c>L</c> such that <c>Σ = LLᵀ</c>.
    /// <para>平均ベクトルと下三角コレスキー因子 <c>L</c>（<c>Σ = LLᵀ</c>）から <see cref="MultivariateNormalDistribution"/> を生成します。</para>
    /// </summary>
    /// <param name="mean">The finite mean vector <c>μ</c> (length at least 1). / 長さ 1 以上の有限な平均ベクトル <c>μ</c>。</param>
    /// <param name="lowerFactor">The lower-triangular Cholesky factor <c>L</c> with positive diagonal entries; the strictly upper triangle is ignored. / 対角成分が正の下三角コレスキー因子 <c>L</c>（狭義上三角は無視されます）。</param>
    /// <returns>A multivariate normal distribution with covariance <c>LLᵀ</c>. / 共分散行列 <c>LLᵀ</c> を持つ多変量正規分布。</returns>
    /// <exception cref="ArgumentException"><paramref name="mean"/> is empty, or <paramref name="lowerFactor"/> has a mismatched shape. / <paramref name="mean"/> が空、または <paramref name="lowerFactor"/> の形状が一致しない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="mean"/> or the lower triangle of <paramref name="lowerFactor"/> contains non-finite entries, a diagonal entry is non-positive, or <c>LLᵀ</c> overflows. / <paramref name="mean"/> または <paramref name="lowerFactor"/> の下三角に非有限の要素がある、対角成分が正でない、または <c>LLᵀ</c> がオーバーフローする場合。</exception>
    public static MultivariateNormalDistribution FromCholeskyFactor(ReadOnlyVectorSpan<double> mean, ReadOnlyMatrixSpan<double> lowerFactor)
    {
        var m = MultivariateMath.FiniteVector(mean, nameof(mean));
        int d = m.Length;
        var factor = MultivariateMath.LowerFactor(lowerFactor, d, nameof(lowerFactor));
        var covariance = MultivariateMath.MultiplyByTranspose(factor, d, nameof(lowerFactor));
        return new MultivariateNormalDistribution(m, covariance, factor);
    }

    /// <inheritdoc/>
    public int Dimension => _mean.Length;

    /// <summary>
    /// Gets a read-only view of the mean vector <c>μ</c>.
    /// <para>平均ベクトル <c>μ</c> の読み取り専用ビューを取得します。</para>
    /// </summary>
    public ReadOnlyVectorSpan<double> Mean => new(_mean);

    /// <summary>
    /// Gets a read-only view of the symmetric covariance matrix <c>Σ</c>.
    /// <para>対称共分散行列 <c>Σ</c> の読み取り専用ビューを取得します。</para>
    /// </summary>
    public ReadOnlyMatrixSpan<double> Covariance => new(_covariance, Dimension, Dimension);

    /// <summary>
    /// Gets a read-only view of the lower-triangular Cholesky factor <c>L</c> (with zeros in the strictly upper triangle).
    /// <para>共分散行列の下三角コレスキー因子 <c>L</c>（狭義上三角は 0）の読み取り専用ビューを取得します。</para>
    /// </summary>
    public ReadOnlyMatrixSpan<double> CholeskyFactor => new(_factor, Dimension, Dimension);

    /// <summary>
    /// Gets the natural logarithm of the covariance determinant, <c>ln det Σ = 2 Σᵢ ln lᵢᵢ</c>.
    /// <para>共分散行列の行列式の自然対数 <c>ln det Σ = 2 Σᵢ ln lᵢᵢ</c> を取得します。</para>
    /// </summary>
    public double LogDeterminant => 2 * _halfLogDeterminant;

    /// <inheritdoc/>
    public Vector<double> GetMean() => new(_mean);

    /// <inheritdoc/>
    public Matrix<double> GetCovariance() => Matrix<double>.FromColumnMajor(Dimension, Dimension, _covariance);

    /// <inheritdoc/>
    public double Pdf(ReadOnlyVectorSpan<double> x) => Math.Exp(LogPdf(x));

    /// <inheritdoc/>
    public double LogPdf(ReadOnlyVectorSpan<double> x)
    {
        RequireLength(x.Length, nameof(x));
        int d = Dimension;
        Span<double> z = d <= MultivariateMath.StackLimit ? stackalloc double[d] : new double[d];
        return LogPdfCore(x, z);
    }

    /// <inheritdoc/>
    public void LogPdf(ReadOnlyMatrixSpan<double> x, VectorSpan<double> destination)
    {
        RequireLength(x.Rows, nameof(x));
        if (destination.Length != x.Columns)
        {
            throw new ArgumentException("The length must equal the number of points.", nameof(destination));
        }

        MultivariateMath.RequireDisjoint(x, destination);
        int d = Dimension;
        Span<double> z = d <= MultivariateMath.StackLimit ? stackalloc double[d] : new double[d];
        for (int j = 0; j < x.Columns; j++)
        {
            destination[j] = LogPdfCore(x.Column(j), z);
        }
    }

    private double LogPdfCore(ReadOnlyVectorSpan<double> x, Span<double> z)
    {
        double status = MultivariateMath.Standardize(x, _mean, _factor, z);
        if (status != 0)
        {
            return double.IsNaN(status) ? double.NaN : double.NegativeInfinity;
        }

        return -Dimension * XMath.Constants.LnSqrtTwoPi - _halfLogDeterminant - MultivariateMath.HalfSquaredNorm(z);
    }

    /// <summary>
    /// Computes the hyperrectangle probability <c>P(lower &lt; X ≤ upper)</c> via numerical integration.
    /// <para>指定した超直方体に入る確率 <c>P(lower &lt; X ≤ upper)</c> を数値積分で計算します。</para>
    /// </summary>
    /// <param name="lower">The lower bounds per coordinate (may include <c>±∞</c>). / 各座標の下端（<c>±∞</c> 指定可）。</param>
    /// <param name="upper">The upper bounds per coordinate (may include <c>±∞</c>). / 各座標の上端（<c>±∞</c> 指定可）。</param>
    /// <param name="options">Optional integration tolerances and evaluation limit (<see langword="null"/> uses absolute tolerance <c>1e-8</c> and maximum <c>2²²</c> evaluations). / 許容誤差および評価回数上限（<see langword="null"/> の場合は絶対許容誤差 <c>1e-8</c>、最大評価回数 <c>2²²</c>）。</param>
    /// <returns>An <see cref="IntegrationResult"/> containing the probability, error estimate (standard error for quasi-Monte Carlo), evaluation count, and status. Returns <see cref="double.NaN"/> if any bound is <see cref="double.NaN"/>, or 0 if any lower bound is greater than or equal to the corresponding upper bound. / 確率、誤差推定値（準モンテカルロ積分時は標準誤差）、評価回数、終了理由を格納した <see cref="IntegrationResult"/>。端が <see cref="double.NaN"/> の場合は確率が <see cref="double.NaN"/>、下端が上端以上の成分がある場合は 0。</returns>
    /// <remarks>
    /// <para>
    /// Results are deterministic for identical inputs (using a fixed seed when quasi-Monte Carlo integration is selected). Relative accuracy may degrade for extremely small tail probabilities.
    /// </para>
    /// <para>
    /// 同一の入力に対して決定論的に同じ結果を返します（準モンテカルロ積分には固定シードを使用）。極小の確率では相対精度が低下することがあります。
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">The length of <paramref name="lower"/> or <paramref name="upper"/> does not equal <see cref="Dimension"/>. / <paramref name="lower"/> または <paramref name="upper"/> の長さが <see cref="Dimension"/> と一致しない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="options"/> specifies negative/non-finite tolerances or a maximum evaluation count smaller than the initial rule size. / <paramref name="options"/> の許容誤差が負・非有限、または最大評価回数が初期点数未満の場合。</exception>
    public IntegrationResult Probability(ReadOnlyVectorSpan<double> lower, ReadOnlyVectorSpan<double> upper, IntegrationOptions? options = null)
    {
        RequireLength(lower.Length, nameof(lower));
        RequireLength(upper.Length, nameof(upper));
        return GenzProbability.Probability(_covariance, _mean, double.PositiveInfinity, lower, upper, options);
    }

    /// <summary>
    /// Computes the multivariate cumulative distribution function <c>P(X ≤ upper)</c> via numerical integration.
    /// <para>指定した点までの多変量累積確率 <c>P(X ≤ upper)</c> を数値積分で計算します。</para>
    /// </summary>
    /// <param name="upper">The upper bounds per coordinate (may include <c>±∞</c>). / 各座標の上端（<c>±∞</c> 指定可）。</param>
    /// <param name="options">Optional integration tolerances and evaluation limit (<see langword="null"/> uses defaults). / 許容誤差および評価回数上限（<see langword="null"/> の場合は既定値）。</param>
    /// <returns>An <see cref="IntegrationResult"/> containing <c>P(X ≤ upper)</c>, error estimate (standard error for quasi-Monte Carlo), evaluation count, and status. Returns <see cref="double.NaN"/> if any bound is <see cref="double.NaN"/>. / 確率 <c>P(X ≤ upper)</c>、誤差推定値（準モンテカルロ積分時は標準誤差）、評価回数、終了理由を格納した <see cref="IntegrationResult"/>。端に <see cref="double.NaN"/> を含む場合は <see cref="double.NaN"/>。</returns>
    /// <exception cref="ArgumentException">The length of <paramref name="upper"/> does not equal <see cref="Dimension"/>. / <paramref name="upper"/> の長さが <see cref="Dimension"/> と一致しない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="options"/> specifies negative/non-finite tolerances or a maximum evaluation count smaller than the initial rule size. / <paramref name="options"/> の許容誤差が負・非有限、または最大評価回数が初期点数未満の場合。</exception>
    public IntegrationResult Cdf(ReadOnlyVectorSpan<double> upper, IntegrationOptions? options = null)
    {
        RequireLength(upper.Length, nameof(upper));
        return GenzProbability.Cdf(_covariance, _mean, double.PositiveInfinity, upper, options);
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, VectorSpan<double> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        RequireLength(destination.Length, nameof(destination));
        int d = Dimension;
        Span<double> z = d <= MultivariateMath.StackLimit ? stackalloc double[d] : new double[d];
        SampleCore(random, z, destination);
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, MatrixSpan<double> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        RequireLength(destination.Rows, nameof(destination));
        int d = Dimension;
        Span<double> z = d <= MultivariateMath.StackLimit ? stackalloc double[d] : new double[d];
        for (int j = 0; j < destination.Columns; j++)
        {
            SampleCore(random, z, destination.Column(j));
        }
    }

    private void SampleCore(IRandomSource random, Span<double> z, VectorSpan<double> destination)
    {
        for (int i = 0; i < z.Length; i++)
        {
            z[i] = Ziggurat.StandardNormal(random);
        }

        MultivariateMath.AddLowerProduct(_mean, _factor, z, destination);
    }

    private void RequireLength(int length, string name)
    {
        if (length != Dimension)
        {
            throw new ArgumentException("The length must equal the dimension.", name);
        }
    }
}
