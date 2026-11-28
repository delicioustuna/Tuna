using Tuna.Integration;
using Tuna.LinearAlgebra;
using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the multivariate Student's t-distribution <c>X = μ + Lz √(ν/V)</c> with degrees of freedom <c>ν</c>, location <c>μ</c>, and scale matrix <c>S = LLᵀ</c>.
/// <para>自由度 <c>ν</c>、位置 <c>μ</c>、尺度行列 <c>S = LLᵀ</c> を持つ多変量 t 分布（<c>X = μ + Lz √(ν/V)</c>）を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Setting <see cref="DegreesOfFreedom"/> to <see cref="double.PositiveInfinity"/> yields <c>N(μ, S)</c>, matching <see cref="StudentTDistribution"/>.
/// <see cref="Scale"/> is the scale matrix <c>S</c>, not the covariance: <see cref="GetMean"/> returns <c>μ</c> for <c>ν &gt; 1</c> (otherwise all <see cref="double.NaN"/>);
/// <see cref="GetCovariance"/> returns <c>ν/(ν - 2) · S</c> for <c>ν &gt; 2</c>, <see cref="double.PositiveInfinity"/> on the diagonal and <see cref="double.NaN"/> off-diagonal for <c>1 &lt; ν ≤ 2</c>, and all <see cref="double.NaN"/> for <c>ν ≤ 1</c>.
/// Parameter validation and evaluation-point handling follow <see cref="MultivariateNormalDistribution"/>; in 1 dimension, <see cref="FromCholeskyFactor"/> matches <see cref="StudentTDistribution"/> samples.
/// </para>
/// <para>
/// <see cref="DegreesOfFreedom"/> に <see cref="double.PositiveInfinity"/> を指定すると多変量正規分布 <c>N(μ, S)</c> として振る舞います（1 変量の <see cref="StudentTDistribution"/> と同じ振る舞い）。
/// <see cref="Scale"/> は尺度行列 <c>S</c> であり共分散ではありません。<see cref="GetMean"/> は <c>ν &gt; 1</c> で <c>μ</c>（それ以外は全成分 <see cref="double.NaN"/>）、
/// <see cref="GetCovariance"/> は <c>ν &gt; 2</c> で <c>ν/(ν - 2) · S</c>、<c>1 &lt; ν ≤ 2</c> で対角成分 <see cref="double.PositiveInfinity"/>・非対角成分 <see cref="double.NaN"/>、<c>ν ≤ 1</c> で全成分 <see cref="double.NaN"/> を返します。
/// パラメーター検証および評価点の扱いは <see cref="MultivariateNormalDistribution"/> と同じであり、1 次元で <see cref="FromCholeskyFactor"/> を用いると <see cref="StudentTDistribution"/> と同一の標本を生成します。
/// </para>
/// </remarks>
public sealed class MultivariateStudentTDistribution : IMultivariateContinuousDistribution
{
    private readonly double _degreesOfFreedom;
    private readonly double _logDegreesOfFreedom;
    private readonly double[] _location;
    private readonly double[] _scale;
    private readonly double[] _factor;
    private readonly double _halfLogDeterminant;
    private readonly double _logNormalization;

    /// <summary>
    /// Initializes a new instance of <see cref="MultivariateStudentTDistribution"/> with the specified degrees of freedom, location vector, and scale matrix.
    /// <para>自由度、位置ベクトル、および尺度行列を指定して <see cref="MultivariateStudentTDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="degreesOfFreedom">The degrees of freedom <c>ν</c> (positive, or <see cref="double.PositiveInfinity"/> for the multivariate normal limit). / 自由度 <c>ν</c>（正、または多変量正規極限として <see cref="double.PositiveInfinity"/>）。</param>
    /// <param name="location">The finite location vector <c>μ</c> (length at least 1). / 長さ 1 以上の有限な位置ベクトル <c>μ</c>。</param>
    /// <param name="scale">The finite <c>d × d</c> symmetric positive definite scale matrix <c>S</c> (lower triangle used). / 有限な <c>d × d</c> 対称正定値の尺度行列 <c>S</c>（下三角を使用）。</param>
    /// <exception cref="ArgumentException"><paramref name="location"/> is empty, or <paramref name="scale"/> has a mismatched shape, is asymmetric, or is not numerically positive definite. / <paramref name="location"/> が空、または <paramref name="scale"/> の形状不一致・非対称・非正定値の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="degreesOfFreedom"/> is not positive, or <paramref name="location"/> or <paramref name="scale"/> contains non-finite entries. / <paramref name="degreesOfFreedom"/> が正でない、または <paramref name="location"/> もしくは <paramref name="scale"/> に非有限の要素が含まれる場合。</exception>
    public MultivariateStudentTDistribution(double degreesOfFreedom, ReadOnlyVectorSpan<double> location, ReadOnlyMatrixSpan<double> scale)
    {
        DistributionArgument.PositiveOrInfinity(degreesOfFreedom, nameof(degreesOfFreedom));
        _location = MultivariateMath.FiniteVector(location, nameof(location));
        int d = _location.Length;
        _scale = MultivariateMath.SymmetricMatrix(scale, d, nameof(scale));
        _factor = MultivariateMath.PositiveDefiniteFactor(_scale, d, nameof(scale));
        _degreesOfFreedom = degreesOfFreedom;
        (_logDegreesOfFreedom, _halfLogDeterminant, _logNormalization) = Constants(degreesOfFreedom, _factor, d);
    }

    private MultivariateStudentTDistribution(double degreesOfFreedom, double[] location, double[] scale, double[] factor)
    {
        _degreesOfFreedom = degreesOfFreedom;
        _location = location;
        _scale = scale;
        _factor = factor;
        (_logDegreesOfFreedom, _halfLogDeterminant, _logNormalization) = Constants(degreesOfFreedom, factor, location.Length);
    }

    /// <summary>
    /// Creates a <see cref="MultivariateStudentTDistribution"/> from degrees of freedom, location vector, and lower-triangular Cholesky factor <c>L</c> such that <c>S = LLᵀ</c>.
    /// <para>自由度、位置ベクトル、および下三角コレスキー因子 <c>L</c>（<c>S = LLᵀ</c>）から <see cref="MultivariateStudentTDistribution"/> を生成します。</para>
    /// </summary>
    /// <param name="degreesOfFreedom">The degrees of freedom <c>ν</c> (positive, or <see cref="double.PositiveInfinity"/> for the multivariate normal limit). / 自由度 <c>ν</c>（正、または多変量正規極限として <see cref="double.PositiveInfinity"/>）。</param>
    /// <param name="location">The finite location vector <c>μ</c> (length at least 1). / 長さ 1 以上の有限な位置ベクトル <c>μ</c>。</param>
    /// <param name="lowerFactor">The lower-triangular Cholesky factor <c>L</c> with positive diagonal entries; the strictly upper triangle is ignored. / 対角成分が正の下三角コレスキー因子 <c>L</c>（狭義上三角は無視されます）。</param>
    /// <returns>A multivariate Student's t-distribution with scale matrix <c>LLᵀ</c>. / 尺度行列 <c>LLᵀ</c> を持つ多変量 t 分布。</returns>
    /// <exception cref="ArgumentException"><paramref name="location"/> is empty, or <paramref name="lowerFactor"/> has a mismatched shape. / <paramref name="location"/> が空、または <paramref name="lowerFactor"/> の形状が一致しない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="degreesOfFreedom"/> is not positive, <paramref name="location"/> or the lower triangle of <paramref name="lowerFactor"/> contains non-finite entries, a diagonal entry is non-positive, or <c>LLᵀ</c> overflows. / <paramref name="degreesOfFreedom"/> が正でない、<paramref name="location"/> または <paramref name="lowerFactor"/> の下三角に非有限の要素がある、対角成分が正でない、または <c>LLᵀ</c> がオーバーフローする場合。</exception>
    public static MultivariateStudentTDistribution FromCholeskyFactor(
        double degreesOfFreedom, ReadOnlyVectorSpan<double> location, ReadOnlyMatrixSpan<double> lowerFactor)
    {
        DistributionArgument.PositiveOrInfinity(degreesOfFreedom, nameof(degreesOfFreedom));
        var m = MultivariateMath.FiniteVector(location, nameof(location));
        int d = m.Length;
        var factor = MultivariateMath.LowerFactor(lowerFactor, d, nameof(lowerFactor));
        var scale = MultivariateMath.MultiplyByTranspose(factor, d, nameof(lowerFactor));
        return new MultivariateStudentTDistribution(degreesOfFreedom, m, scale, factor);
    }

    private static (double LogNu, double HalfLogDeterminant, double LogNormalization) Constants(double nu, double[] factor, int d)
    {
        double halfLogDeterminant = MultivariateMath.SumLogDiagonal(factor, d);
        double normalization = double.IsPositiveInfinity(nu)
            ? -d * XMath.Constants.LnSqrtTwoPi
            : DistributionMath.LogGammaShiftCorrection(0.5 * nu, 0.5 * d) - d * XMath.Constants.LnSqrtTwoPi;
        return (Math.Log(nu), halfLogDeterminant, normalization);
    }

    /// <inheritdoc/>
    public int Dimension => _location.Length;

    /// <summary>
    /// Gets the degrees of freedom <c>ν</c>.
    /// <para>自由度 <c>ν</c> を取得します。</para>
    /// </summary>
    public double DegreesOfFreedom => _degreesOfFreedom;

    /// <summary>
    /// Gets a read-only view of the location vector <c>μ</c>.
    /// <para>位置ベクトル <c>μ</c> の読み取り専用ビューを取得します。</para>
    /// </summary>
    public ReadOnlyVectorSpan<double> Location => new(_location);

    /// <summary>
    /// Gets a read-only view of the symmetric scale matrix <c>S</c> (distinct from the covariance matrix).
    /// <para>対称な尺度行列 <c>S</c>（共分散行列とは異なります）の読み取り専用ビューを取得します。</para>
    /// </summary>
    public ReadOnlyMatrixSpan<double> Scale => new(_scale, Dimension, Dimension);

    /// <summary>
    /// Gets a read-only view of the lower-triangular Cholesky factor <c>L</c> of <see cref="Scale"/> (with zeros in the strictly upper triangle).
    /// <para><see cref="Scale"/> の下三角コレスキー因子 <c>L</c>（狭義上三角は 0）の読み取り専用ビューを取得します。</para>
    /// </summary>
    public ReadOnlyMatrixSpan<double> CholeskyFactor => new(_factor, Dimension, Dimension);

    private bool IsNormal => double.IsPositiveInfinity(_degreesOfFreedom);

    /// <inheritdoc/>
    public Vector<double> GetMean()
    {
        var mean = new Vector<double>(_location);
        if (!(_degreesOfFreedom > 1))
        {
            mean.AsSpan().Fill(double.NaN);
        }

        return mean;
    }

    /// <inheritdoc/>
    public Matrix<double> GetCovariance()
    {
        int d = Dimension;
        double nu = _degreesOfFreedom;
        var covariance = Matrix<double>.FromColumnMajor(d, d, _scale);
        if (nu > 2)
        {
            if (!IsNormal)
            {
                double factor = nu / (nu - 2);
                var data = covariance.AsSpan();
                for (int i = 0; i < data.Length; i++)
                {
                    data[i] *= factor;
                }
            }

            return covariance;
        }

        covariance.AsSpan().Fill(double.NaN);
        if (nu > 1)
        {
            covariance.Diagonal().Fill(double.PositiveInfinity);
        }

        return covariance;
    }

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
        double status = MultivariateMath.Standardize(x, _location, _factor, z);
        if (status != 0)
        {
            return double.IsNaN(status) ? double.NaN : double.NegativeInfinity;
        }

        if (IsNormal)
        {
            return _logNormalization - _halfLogDeterminant - MultivariateMath.HalfSquaredNorm(z);
        }

        return _logNormalization - 0.5 * (_degreesOfFreedom + Dimension) * LogOnePlusRatio(z) - _halfLogDeterminant;
    }

    // Overflow-safe ln(1 + ||z||² / ν).
    private double LogOnePlusRatio(ReadOnlySpan<double> z)
    {
        double squaredNorm = 0;
        foreach (double value in z)
        {
            squaredNorm += value * value;
        }

        double ratio = squaredNorm / _degreesOfFreedom;
        if (ratio < 1e300)
        {
            return ratio > 1 ? Math.Log(ratio) + DistributionMath.Log1p(1 / ratio) : DistributionMath.Log1p(ratio);
        }

        // For ||z||²/ν >= 1e300, ln(1 + ν/||z||²) < ε; rescale by max component if ||z||² overflows.
        double logSquaredNorm;
        if (squaredNorm < double.PositiveInfinity)
        {
            logSquaredNorm = Math.Log(squaredNorm);
        }
        else
        {
            double largest = 0;
            foreach (double value in z)
            {
                largest = Math.Max(largest, Math.Abs(value));
            }

            double scaled = 0;
            foreach (double value in z)
            {
                double r = value / largest;
                scaled += r * r;
            }

            logSquaredNorm = 2 * Math.Log(largest) + Math.Log(scaled);
        }

        return logSquaredNorm - _logDegreesOfFreedom;
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
        return GenzProbability.Probability(_scale, _location, _degreesOfFreedom, lower, upper, options);
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
        return GenzProbability.Cdf(_scale, _location, _degreesOfFreedom, upper, options);
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, VectorSpan<double> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        RequireLength(destination.Length, nameof(destination));
        int d = Dimension;
        Span<double> t = d <= MultivariateMath.StackLimit ? stackalloc double[d] : new double[d];
        SampleCore(random, t, destination);
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, MatrixSpan<double> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        RequireLength(destination.Rows, nameof(destination));
        int d = Dimension;
        Span<double> t = d <= MultivariateMath.StackLimit ? stackalloc double[d] : new double[d];
        for (int j = 0; j < destination.Columns; j++)
        {
            SampleCore(random, t, destination.Column(j));
        }
    }

    private void SampleCore(IRandomSource random, Span<double> t, VectorSpan<double> destination)
    {
        for (int i = 0; i < t.Length; i++)
        {
            t[i] = Ziggurat.StandardNormal(random);
        }

        if (!IsNormal)
        {
            // V = 2G ~ χ²(ν), w = sqrt(ν / V).
            double logV = XMath.Constants.Ln2 + Sampling.LogStandardGamma(random, 0.5 * _degreesOfFreedom);
            double w = Math.Exp(0.5 * (_logDegreesOfFreedom - logV));
            bool finite = true;
            for (int i = 0; i < t.Length; i++)
            {
                finite &= double.IsFinite(t[i] * w);
            }

            if (!finite)
            {
                // When z*w overflows for tiny ν, compute Lz first and scale by w to avoid 0*∞ or ∞ - ∞.
                MultivariateMath.AddScaledLowerProduct(_location, _factor, t, w, destination);
                return;
            }

            for (int i = 0; i < t.Length; i++)
            {
                t[i] *= w;
            }
        }

        MultivariateMath.AddLowerProduct(_location, _factor, t, destination);
    }

    private void RequireLength(int length, string name)
    {
        if (length != Dimension)
        {
            throw new ArgumentException("The length must equal the dimension.", name);
        }
    }
}
