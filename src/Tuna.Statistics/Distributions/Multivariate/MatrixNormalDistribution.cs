using Tuna.LinearAlgebra;
using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the matrix normal distribution <c>MN(M, U, V)</c> with row covariance <c>U</c> and column covariance <c>V</c>.
/// <para>行共分散 <c>U</c> と列共分散 <c>V</c> を持つ行列正規分布 <c>MN(M, U, V)</c> を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Vectorizing column-by-column satisfies <c>vec(X) ~ N(vec(M), V ⊗ U)</c>. Note that <c>(cU, V/c)</c> defines the same distribution for any <c>c &gt; 0</c>.
/// Covariance validation and evaluation-point handling follow <see cref="MultivariateNormalDistribution"/>.
/// </para>
/// <para>
/// 列順にベクトル化した <c>vec(X)</c> は <c>N(vec(M), V ⊗ U)</c> に従います。任意の <c>c &gt; 0</c> に対して <c>(cU, V/c)</c> は同一の分布を表します。
/// 共分散行列の検証および評価点の扱いは <see cref="MultivariateNormalDistribution"/> と同じです。
/// </para>
/// </remarks>
public sealed class MatrixNormalDistribution : IMatrixContinuousDistribution
{
    private readonly int _rows;
    private readonly int _columns;
    private readonly double[] _mean;
    private readonly double[] _rowCovariance;
    private readonly double[] _columnCovariance;
    private readonly double[] _rowFactor;
    private readonly double[] _columnFactor;
    private readonly double _logNormalization;

    /// <summary>
    /// Initializes a new instance of <see cref="MatrixNormalDistribution"/> with the specified mean, row covariance, and column covariance.
    /// <para>平均行列、行共分散行列、および列共分散行列を指定して <see cref="MatrixNormalDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="mean">The finite <c>n × p</c> (<c>n, p ≥ 1</c>) mean matrix <c>M</c>. / 有限な <c>n × p</c>（<c>n, p ≥ 1</c>）の平均行列 <c>M</c>。</param>
    /// <param name="rowCovariance">The finite <c>n × n</c> symmetric positive definite row covariance matrix <c>U</c> (lower triangle used). / 有限な <c>n × n</c> 対称正定値の行共分散行列 <c>U</c>（下三角を使用）。</param>
    /// <param name="columnCovariance">The finite <c>p × p</c> symmetric positive definite column covariance matrix <c>V</c> (lower triangle used). / 有限な <c>p × p</c> 対称正定値の列共分散行列 <c>V</c>（下三角を使用）。</param>
    /// <exception cref="ArgumentException"><paramref name="mean"/> is empty, or either covariance matrix has a mismatched shape, is asymmetric, or is not numerically positive definite. / <paramref name="mean"/> が空、または共分散行列の形状不一致・非対称・非正定値の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">Any matrix contains non-finite entries. / いずれかの行列に非有限の要素が含まれる場合。</exception>
    public MatrixNormalDistribution(ReadOnlyMatrixSpan<double> mean, ReadOnlyMatrixSpan<double> rowCovariance, ReadOnlyMatrixSpan<double> columnCovariance)
    {
        int n = mean.Rows;
        int p = mean.Columns;
        if (n < 1 || p < 1)
        {
            throw new ArgumentException("The mean must have at least one row and one column.", nameof(mean));
        }

        _mean = MultivariateMath.FiniteMatrix(mean, n, p, nameof(mean));
        _rowCovariance = MultivariateMath.SymmetricMatrix(rowCovariance, n, nameof(rowCovariance));
        _columnCovariance = MultivariateMath.SymmetricMatrix(columnCovariance, p, nameof(columnCovariance));
        _rowFactor = MultivariateMath.PositiveDefiniteFactor(_rowCovariance, n, nameof(rowCovariance));
        _columnFactor = MultivariateMath.PositiveDefiniteFactor(_columnCovariance, p, nameof(columnCovariance));
        _rows = n;
        _columns = p;
        _logNormalization = -((double)n * p) * DistributionMath.LnSqrt2Pi
            - p * MultivariateMath.SumLogDiagonal(_rowFactor, n) - n * MultivariateMath.SumLogDiagonal(_columnFactor, p);
    }

    /// <inheritdoc/>
    public int Rows => _rows;

    /// <inheritdoc/>
    public int Columns => _columns;

    /// <summary>
    /// Gets a read-only view of the mean matrix <c>M</c>.
    /// <para>平均行列 <c>M</c> の読み取り専用ビューを取得します。</para>
    /// </summary>
    public ReadOnlyMatrixSpan<double> Mean => new(_mean, _rows, _columns);

    /// <summary>
    /// Gets a read-only view of the symmetric row covariance matrix <c>U</c>.
    /// <para>対称な行共分散行列 <c>U</c> の読み取り専用ビューを取得します。</para>
    /// </summary>
    public ReadOnlyMatrixSpan<double> RowCovariance => new(_rowCovariance, _rows, _rows);

    /// <summary>
    /// Gets a read-only view of the symmetric column covariance matrix <c>V</c>.
    /// <para>対称な列共分散行列 <c>V</c> の読み取り専用ビューを取得します。</para>
    /// </summary>
    public ReadOnlyMatrixSpan<double> ColumnCovariance => new(_columnCovariance, _columns, _columns);

    /// <inheritdoc/>
    public Matrix<double> GetMean() => Matrix<double>.FromColumnMajor(_rows, _columns, _mean);

    /// <inheritdoc/>
    public double Pdf(ReadOnlyMatrixSpan<double> x) => Math.Exp(LogPdf(x));

    /// <inheritdoc/>
    public double LogPdf(ReadOnlyMatrixSpan<double> x)
    {
        int n = _rows;
        int p = _columns;
        MultivariateMath.RequireShape(x, n, p, nameof(x));
        double status = MultivariateMath.NonFiniteStatus(x);
        if (status != 0)
        {
            return double.IsNaN(status) ? double.NaN : double.NegativeInfinity;
        }

        int length = _mean.Length;
        Span<double> z = length <= MultivariateMath.StackLimit ? stackalloc double[length] : new double[length];
        bool halved = false;
        for (int j = 0; j < p && !halved; j++)
        {
            for (int i = 0; i < n; i++)
            {
                double difference = x[i, j] - _mean[i + j * n];
                if (double.IsInfinity(difference))
                {
                    halved = true;
                    break;
                }

                z[i + j * n] = difference;
            }
        }

        if (halved)
        {
            for (int j = 0; j < p; j++)
            {
                for (int i = 0; i < n; i++)
                {
                    z[i + j * n] = 0.5 * x[i, j] - 0.5 * _mean[i + j * n];
                }
            }
        }

        // Solve Z <- L_U⁻¹ Z by column, then Z <- Z L_V⁻ᵀ by row.
        for (int j = 0; j < p; j++)
        {
            MultivariateMath.SolveLowerInPlace(_rowFactor, n, z, offset: j * n);
        }

        for (int i = 0; i < n; i++)
        {
            MultivariateMath.SolveLowerInPlace(_columnFactor, p, z, offset: i, stride: n);
        }

        double half = 0;
        foreach (double value in z)
        {
            double scaled = halved ? value * 2 : value;
            if (!double.IsFinite(scaled))
            {
                return double.NegativeInfinity;
            }

            half += 0.5 * scaled * scaled;
        }

        return _logNormalization - half;
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, MatrixSpan<double> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        int n = _rows;
        int p = _columns;
        MultivariateMath.RequireShape(destination, n, p, nameof(destination));
        int length = _mean.Length;
        Span<double> y = length <= MultivariateMath.StackLimit ? stackalloc double[length] : new double[length];
        for (int i = 0; i < y.Length; i++)
        {
            y[i] = Ziggurat.StandardNormal(random);
        }

        // In-place Y = L_U Z; iterating rows bottom-up preserves unread upper entries.
        for (int j = 0; j < p; j++)
        {
            for (int i = n - 1; i >= 0; i--)
            {
                double sum = 0;
                for (int k = 0; k <= i; k++)
                {
                    sum += _rowFactor[i + k * n] * y[k + j * n];
                }

                y[i + j * n] = sum;
            }
        }

        // X = M + Y L_Vᵀ; halve M and L_V and double back if intermediate summation overflows while Y is finite.
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
            {
                double sum = 0;
                bool finite = true;
                for (int k = 0; k <= j; k++)
                {
                    double value = y[i + k * n];
                    sum += value * _columnFactor[j + k * p];
                    finite &= double.IsFinite(value);
                }

                double result = _mean[i + j * n] + sum;
                if (double.IsInfinity(result) && finite)
                {
                    double half = 0;
                    for (int k = 0; k <= j; k++)
                    {
                        half += y[i + k * n] * (0.5 * _columnFactor[j + k * p]);
                    }

                    result = 2 * (0.5 * _mean[i + j * n] + half);
                }

                destination[i, j] = result;
            }
        }
    }
}
