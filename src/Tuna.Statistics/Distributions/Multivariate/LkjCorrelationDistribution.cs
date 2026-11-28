using Tuna.LinearAlgebra;
using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the Lewandowski–Kurowicka–Joe (LKJ) distribution over <c>d × d</c> correlation matrices.
/// <para><c>d × d</c> 相関行列上の LKJ（Lewandowski–Kurowicka–Joe）分布を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Density is proportional to <c>det(R)^(η - 1)</c> with respect to Lebesgue measure on the <c>d(d - 1)/2</c> strictly lower-triangular entries (<c>η = 1</c> is uniform over correlation matrices, <c>η &gt; 1</c> concentrates near the identity, and <c>η &lt; 1</c> favors stronger correlations).
/// Evaluation returns <see cref="double.NaN"/> if the input contains <see cref="double.NaN"/>, and treats matrices as outside the support (density 0) if they contain infinities, are asymmetric beyond tolerance, have diagonal entries differing from 1 by more than <c>√ε</c>, or are not numerically positive definite.
/// Density evaluation treats diagonal entries as 1; sampling produces correlation matrices with exact 1s on the diagonal.
/// </para>
/// <para>
/// 密度は対角を除く下三角の <c>d(d - 1)/2</c> 成分に対するルベーグ測度に関して <c>det(R)^(η - 1)</c> に比例します（<c>η = 1</c> で相関行列上一様、<c>η &gt; 1</c> で単位行列付近に集中、<c>η &lt; 1</c> で強い相関に集中します）。
/// 評価点に <see cref="double.NaN"/> を含む場合は <see cref="double.NaN"/> を返し、無限大を含む場合、許容誤差を超えて非対称な場合、対角成分が 1 から <c>√ε</c> を超えて離れている場合、または数値的に正定値でない場合は台の外（密度 0）として扱います。
/// 密度計算では対角成分を 1 とみなし、サンプリングでは対角成分が厳密に 1 の相関行列を生成します。
/// </para>
/// </remarks>
public sealed partial class LkjCorrelationDistribution : IMatrixContinuousDistribution
{
    private readonly int _dimension;
    private readonly double _shape;
    private readonly double _logNormalization;

    /// <summary>
    /// Initializes a new instance of <see cref="LkjCorrelationDistribution"/> with the specified dimension and shape parameter.
    /// <para>次元と形状パラメーターを指定して <see cref="LkjCorrelationDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="dimension">The dimension <c>d</c> of the correlation matrix (<c>d ≥ 2</c>). / 相関行列の次元 <c>d</c>（2 以上）。</param>
    /// <param name="shape">The shape parameter <c>η</c> (positive and finite). / 形状パラメーター <c>η</c>（正かつ有限）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="dimension"/> is less than 2, or <paramref name="shape"/> is not positive and finite. / <paramref name="dimension"/> が 2 未満、または <paramref name="shape"/> が正の有限値でない場合。</exception>
    public LkjCorrelationDistribution(int dimension, double shape)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(dimension, 2);
        DistributionArgument.PositiveFinite(shape, nameof(shape));
        _dimension = dimension;
        _shape = shape;
        _logNormalization = LogNormalization(dimension, shape);
    }

    private static double LogNormalization(int dimension, double shape)
    {
        double logC = 0;
        for (int k = 1; k < dimension; k++)
        {
            double beta = shape + 0.5 * (dimension - k - 1);
            logC += (dimension - k) * (0.5 * XMath.Constants.LnPi - 0.5 * Math.Log(beta) - DistributionMath.LogGammaHalfStepCorrection(beta));
        }

        return -logC;
    }

    /// <summary>
    /// Gets the dimension <c>d</c> of the correlation matrix.
    /// <para>相関行列の次元 <c>d</c> を取得します。</para>
    /// </summary>
    public int Dimension => _dimension;

    int IMatrixContinuousDistribution.Rows => _dimension;

    int IMatrixContinuousDistribution.Columns => _dimension;

    /// <summary>
    /// Gets the shape parameter <c>η</c>.
    /// <para>形状パラメーター <c>η</c> を取得します。</para>
    /// </summary>
    public double Shape => _shape;

    /// <inheritdoc/>
    public Matrix<double> GetMean() => Matrix<double>.Identity(_dimension);

    /// <inheritdoc/>
    public double Pdf(ReadOnlyMatrixSpan<double> x) => Math.Exp(LogPdf(x));

    /// <inheritdoc/>
    public double LogPdf(ReadOnlyMatrixSpan<double> x)
    {
        int d = _dimension;
        MultivariateMath.RequireShape(x, d, d, nameof(x));
        double status = MultivariateMath.NonFiniteStatus(x);
        if (status != 0)
        {
            return double.IsNaN(status) ? double.NaN : double.NegativeInfinity;
        }

        if (!MultivariateMath.IsSymmetric(x))
        {
            return double.NegativeInfinity;
        }

        for (int i = 0; i < d; i++)
        {
            if (!(Math.Abs(x[i, i] - 1) <= MultivariateMath.SymmetryTolerance))
            {
                return double.NegativeInfinity;
            }
        }

        int length = d * d;
        Span<double> c = length <= MultivariateMath.StackLimit ? stackalloc double[length] : new double[length];
        MultivariateMath.CopyLowerSymmetric(x, c, d);
        for (int i = 0; i < d; i++)
        {
            c[i + i * d] = 1;
        }

        if (!MultivariateMath.FactorInPlace(c, d))
        {
            return double.NegativeInfinity;
        }

        return (_shape - 1) * FactorLogDeterminant(c, d) + _logNormalization;
    }

    private static double FactorLogDeterminant(ReadOnlySpan<double> factor, int d)
    {
        // Since c_jj² = 1 - Σ_{k<j} c_jk², compute ln det R via Log1p for accuracy near the identity.
        double logDeterminant = 0;
        for (int j = 1; j < d; j++)
        {
            double sum = 0;
            for (int k = 0; k < j; k++)
            {
                double value = factor[j + k * d];
                sum += value * value;
            }

            logDeterminant += DistributionMath.Log1p(-sum);
        }

        return logDeterminant;
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, MatrixSpan<double> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        int d = _dimension;
        MultivariateMath.RequireShape(destination, d, d, nameof(destination));
        int length = d * d;
        Span<double> c = length <= MultivariateMath.StackLimit ? stackalloc double[length] : new double[length];
        c[0] = 1;
        for (int k = 1; k < d; k++)
        {
            double squaredNorm;
            do
            {
                squaredNorm = 0;
                for (int j = 0; j < k; j++)
                {
                    double z = Ziggurat.StandardNormal(random);
                    c[k + j * d] = z;
                    squaredNorm += z * z;
                }
            }
            while (squaredNorm == 0);

            // Compute y and 1 - y independently from log-gammas to avoid cancellation near 1.
            double logA = Sampling.LogStandardGamma(random, 0.5 * k);
            double logB = Sampling.LogStandardGamma(random, _shape + 0.5 * (d - k - 1));
            double y = 1 / (1 + Math.Exp(logB - logA));
            double complement = 1 / (1 + Math.Exp(logA - logB));
            double radius = Math.Sqrt(y / squaredNorm);
            for (int j = 0; j < k; j++)
            {
                c[k + j * d] *= radius;
            }

            c[k + k * d] = Math.Sqrt(complement);
        }

        MultivariateMath.WriteLowerGram(c, d, destination);
        for (int i = 0; i < d; i++)
        {
            destination[i, i] = 1;
        }
    }
}
