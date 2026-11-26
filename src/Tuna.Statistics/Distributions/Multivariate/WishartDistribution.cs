using Tuna.LinearAlgebra;
using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the Wishart distribution <c>W_p(ν, Σ)</c> over symmetric positive definite <c>p × p</c> matrices.
/// <para>対称正定値 <c>p × p</c> 行列上のウィシャート分布 <c>W_p(ν, Σ)</c> を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Density is with respect to Lebesgue measure on the <c>p(p + 1)/2</c> lower-triangular entries, with real degrees of freedom <c>ν &gt; p - 1</c>.
/// Evaluation returns <see cref="double.NaN"/> if the input contains <see cref="double.NaN"/>, and 0 (log-density <see cref="double.NegativeInfinity"/>) if it contains infinities, is asymmetric beyond tolerance, or is not numerically positive definite (including singular boundary matrices).
/// Sampling produces exactly symmetric matrices via Bartlett decomposition; for small <c>ν - p + 1</c>, generated matrices may be numerically indistinguishable from singular in <see cref="double"/> precision, in which case <see cref="LogPdf(ReadOnlyMatrixSpan{double})"/> returns <see cref="double.NegativeInfinity"/>.
/// </para>
/// <para>
/// 密度は下三角の <c>p(p + 1)/2</c> 成分に対するルベーグ測度に関して定義され、自由度は実数 <c>ν &gt; p - 1</c> をとります。
/// 評価点に <see cref="double.NaN"/> を含む場合は <see cref="double.NaN"/> を返し、無限大を含む場合、許容誤差を超えて非対称な場合、または数値的に正定値でない場合（特異な境界を含む）は密度 0（対数密度 <see cref="double.NegativeInfinity"/>）を返します。
/// サンプリングはバートレット分解により厳密に対称な行列を生成しますが、<c>ν - p + 1</c> が小さい場合は倍精度で特異と区別できない標本が出ることがあり、その標本の <see cref="LogPdf(ReadOnlyMatrixSpan{double})"/> は <see cref="double.NegativeInfinity"/> になります。
/// </para>
/// </remarks>
public sealed partial class WishartDistribution : IMatrixContinuousDistribution
{
    private readonly int _dimension;
    private readonly double _degreesOfFreedom;
    private readonly double[] _scale;
    private readonly double[] _factor;
    private readonly double _logNormalization;

    /// <summary>
    /// Initializes a new instance of <see cref="WishartDistribution"/> with the specified degrees of freedom and scale matrix.
    /// <para>自由度と尺度行列を指定して <see cref="WishartDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="degreesOfFreedom">The degrees of freedom <c>ν</c> (finite and greater than <c>p - 1</c>). / 自由度 <c>ν</c>（<c>p - 1</c> より大きい有限値）。</param>
    /// <param name="scale">The <c>p × p</c> (<c>p ≥ 1</c>) finite symmetric positive definite scale matrix <c>Σ</c> (lower triangle used). / <c>p × p</c>（<c>p ≥ 1</c>）の有限な対称正定値尺度行列 <c>Σ</c>（下三角を使用）。</param>
    /// <exception cref="ArgumentException"><paramref name="scale"/> is empty, non-square, asymmetric, or not numerically positive definite. / <paramref name="scale"/> が空、非正方、非対称、または数値的に正定値でない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="degreesOfFreedom"/> is not finite or not greater than <c>p - 1</c>, or <paramref name="scale"/> contains non-finite entries. / <paramref name="degreesOfFreedom"/> が <c>p - 1</c> より大きい有限値でない場合、または <paramref name="scale"/> に非有限の要素が含まれる場合。</exception>
    public WishartDistribution(double degreesOfFreedom, ReadOnlyMatrixSpan<double> scale)
    {
        int p = scale.Rows;
        if (p < 1)
        {
            throw new ArgumentException("The dimension must be at least 1.", nameof(scale));
        }

        _scale = MultivariateMath.SymmetricMatrix(scale, p, nameof(scale));
        if (!(degreesOfFreedom > p - 1 && degreesOfFreedom < double.PositiveInfinity))
        {
            throw new ArgumentOutOfRangeException(nameof(degreesOfFreedom), degreesOfFreedom, "The degrees of freedom must be finite and greater than the dimension minus one.");
        }

        _factor = MultivariateMath.PositiveDefiniteFactor(_scale, p, nameof(scale));
        _dimension = p;
        _degreesOfFreedom = degreesOfFreedom;
        double halfLogDeterminant = MultivariateMath.SumLogDiagonal(_factor, p);
        _logNormalization = LogNormalization(p, halfLogDeterminant);
    }

    /// <summary>
    /// Gets the matrix dimension <c>p</c> (samples are <c>p × p</c> matrices).
    /// <para>行列の次元 <c>p</c>（確率変数は <c>p × p</c> 行列）を取得します。</para>
    /// </summary>
    public int Dimension => _dimension;

    int IMatrixContinuousDistribution.Rows => Dimension;

    int IMatrixContinuousDistribution.Columns => Dimension;

    /// <summary>
    /// Gets the degrees of freedom <c>ν</c>.
    /// <para>自由度 <c>ν</c> を取得します。</para>
    /// </summary>
    public double DegreesOfFreedom => _degreesOfFreedom;

    /// <summary>
    /// Gets a read-only view of the symmetric scale matrix <c>Σ</c>.
    /// <para>対称尺度行列 <c>Σ</c> の読み取り専用ビューを取得します。</para>
    /// </summary>
    public ReadOnlyMatrixSpan<double> Scale => new(_scale, Dimension, Dimension);

    /// <inheritdoc/>
    public Matrix<double> GetMean()
    {
        int p = Dimension;
        var mean = Matrix<double>.Uninitialized(p, p);
        var data = mean.AsSpan();
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = _degreesOfFreedom * _scale[i];
        }

        return mean;
    }

    /// <summary>
    /// Computes the elementwise variance matrix of the distribution.
    /// <para>分布の各成分の分散行列を計算します。</para>
    /// </summary>
    /// <returns>A newly allocated <c>p × p</c> matrix containing the elementwise variances <c>Var(Xᵢⱼ) = ν(σᵢⱼ² + σᵢᵢσⱼⱼ)</c>. / 各成分の分散 <c>Var(Xᵢⱼ) = ν(σᵢⱼ² + σᵢᵢσⱼⱼ)</c> を格納した新しい <c>p × p</c> 行列。</returns>
    public Matrix<double> GetVariance()
    {
        int p = Dimension;
        var variance = Matrix<double>.Uninitialized(p, p);
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < p; i++)
            {
                double sij = _scale[i + j * p];
                variance[i, j] = _degreesOfFreedom * (sij * sij + _scale[i + i * p] * _scale[j + j * p]);
            }
        }

        return variance;
    }

    /// <inheritdoc/>
    public double Pdf(ReadOnlyMatrixSpan<double> x) => Math.Exp(LogPdf(x));

    /// <inheritdoc/>
    public double LogPdf(ReadOnlyMatrixSpan<double> x)
    {
        int p = Dimension;
        MultivariateMath.RequireShape(x, p, p, nameof(x));
        double status = MultivariateMath.NonFiniteStatus(x);
        if (status != 0)
        {
            return double.IsNaN(status) ? double.NaN : double.NegativeInfinity;
        }

        if (!MultivariateMath.IsSymmetric(x))
        {
            return double.NegativeInfinity;
        }

        int length = p * p;
        Span<double> a = length <= MultivariateMath.StackLimit ? stackalloc double[length] : new double[length];
        MultivariateMath.CopyLowerSymmetric(x, a, p);
        if (!MultivariateMath.FactorInPlace(a, p))
        {
            return double.NegativeInfinity;
        }

        return FactorLogPdf(_degreesOfFreedom, _factor, p, _logNormalization, a);
    }

    private static double LogNormalization(int p, double halfLogDeterminant) =>
        -(p + 1) * halfLogDeterminant - 0.5 * p * (p + 1.0) * DistributionMath.Ln2
        - 0.25 * p * (p - 1.0) * DistributionMath.LnPi;

    private static double FactorLogPdf(double nu, ReadOnlySpan<double> factor, int p, double logNormalization, Span<double> a)
    {
        // A = L⁻¹C (lower triangular).
        for (int j = 0; j < p; j++)
        {
            MultivariateMath.SolveLowerInPlace(factor, p, a, offset: j * p, start: j);
        }

        double sum = logNormalization;
        for (int j = 0; j < p; j++)
        {
            double k = 0.5 * (nu - j);
            sum += MultivariateMath.LogGammaPrefixOfHalfSquare(k, a[j + j * p], out double logY) - 0.5 * (p + 1 - j) * logY;
            for (int i = j + 1; i < p; i++)
            {
                double value = a[i + j * p];
                sum -= 0.5 * value * value;
            }
        }

        // If tr(Σ⁻¹X) = ||A||² overflows in the extreme tail, ∞ - ∞ yields NaN while the true density is 0.
        return double.IsNaN(sum) ? double.NegativeInfinity : sum;
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, MatrixSpan<double> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        int p = Dimension;
        MultivariateMath.RequireShape(destination, p, p, nameof(destination));
        int length = p * p;
        Span<double> a = length <= MultivariateMath.StackLimit ? stackalloc double[length] : new double[length];
        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < i; j++)
            {
                a[i + j * p] = Ziggurat.StandardNormal(random);
            }

            a[i + i * p] = MultivariateMath.ChiSample(random, 0.5 * (_degreesOfFreedom - i));
        }

        MultivariateMath.MultiplyLowerInPlace(_factor, p, a);
        MultivariateMath.WriteLowerGram(a, p, destination);
    }
}
