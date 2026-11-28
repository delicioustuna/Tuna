using Tuna.LinearAlgebra;
using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the inverse Wishart distribution <c>W⁻¹_p(ν, Ψ)</c> over symmetric positive definite <c>p × p</c> matrices, where <c>X⁻¹ ~ W_p(ν, Ψ⁻¹)</c>.
/// <para>対称正定値 <c>p × p</c> 行列上の逆ウィシャート分布 <c>W⁻¹_p(ν, Ψ)</c>（<c>X⁻¹ ~ W_p(ν, Ψ⁻¹)</c>）を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Density is with respect to Lebesgue measure on the <c>p(p + 1)/2</c> lower-triangular entries, with real degrees of freedom <c>ν &gt; p - 1</c>.
/// For <c>ν ≤ p + 1</c>, <see cref="GetMean"/> returns <see cref="double.PositiveInfinity"/> on the diagonal and <see cref="double.NaN"/> off the diagonal.
/// Evaluation point validation follows <see cref="WishartDistribution"/>; generated samples are exactly symmetric, though small <c>ν - p + 1</c> may produce numerically singular matrices in <see cref="double"/> precision.
/// </para>
/// <para>
/// 密度は下三角の <c>p(p + 1)/2</c> 成分に対するルベーグ測度に関して定義され、自由度は実数 <c>ν &gt; p - 1</c> をとります。
/// <c>ν ≤ p + 1</c> のとき、<see cref="GetMean"/> は対角成分に <see cref="double.PositiveInfinity"/>、非対角成分に <see cref="double.NaN"/> を返します。
/// 評価点の検証条件は <see cref="WishartDistribution"/> と同じです。サンプリングは厳密に対称な行列を返しますが、<c>ν - p + 1</c> が小さい場合は倍精度で特異と区別できない行列が生成されることがあります。
/// </para>
/// </remarks>
public sealed class InverseWishartDistribution : IMatrixContinuousDistribution
{
    private readonly int _dimension;
    private readonly double _degreesOfFreedom;
    private readonly double[] _scale;
    private readonly double[] _factor;
    private readonly double _logNormalization;

    /// <summary>
    /// Initializes a new instance of <see cref="InverseWishartDistribution"/> with the specified degrees of freedom and scale matrix.
    /// <para>自由度と尺度行列を指定して <see cref="InverseWishartDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="degreesOfFreedom">The degrees of freedom <c>ν</c> (finite and greater than <c>p - 1</c>). / 自由度 <c>ν</c>（<c>p - 1</c> より大きい有限値）。</param>
    /// <param name="scale">The <c>p × p</c> (<c>p ≥ 1</c>) finite symmetric positive definite scale matrix <c>Ψ</c>. / <c>p × p</c>（<c>p ≥ 1</c>）の有限な対称正定値尺度行列 <c>Ψ</c>。</param>
    /// <exception cref="ArgumentException"><paramref name="scale"/> is empty, non-square, asymmetric, or not numerically positive definite. / <paramref name="scale"/> が空、非正方、非対称、または数値的に正定値でない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="degreesOfFreedom"/> is not finite or not greater than <c>p - 1</c>, or <paramref name="scale"/> contains non-finite entries. / <paramref name="degreesOfFreedom"/> が <c>p - 1</c> より大きい有限値でない場合、または <paramref name="scale"/> に非有限の要素が含まれる場合。</exception>
    public InverseWishartDistribution(double degreesOfFreedom, ReadOnlyMatrixSpan<double> scale)
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
        _logNormalization = -(p + 1) * halfLogDeterminant + 0.5 * p * (p + 1.0) * XMath.Constants.Ln2
            - 0.25 * p * (p - 1.0) * XMath.Constants.LnPi;
    }

    /// <summary>
    /// Gets the matrix dimension <c>p</c> (samples are <c>p × p</c> matrices).
    /// <para>行列の次元 <c>p</c>（確率変数は <c>p × p</c> 行列）を取得します。</para>
    /// </summary>
    public int Dimension => _dimension;

    int IMatrixContinuousDistribution.Rows => _dimension;

    int IMatrixContinuousDistribution.Columns => _dimension;

    /// <summary>
    /// Gets the degrees of freedom <c>ν</c>.
    /// <para>自由度 <c>ν</c> を取得します。</para>
    /// </summary>
    public double DegreesOfFreedom => _degreesOfFreedom;

    /// <summary>
    /// Gets a read-only view of the symmetric scale matrix <c>Ψ</c>.
    /// <para>対称尺度行列 <c>Ψ</c> の読み取り専用ビューを取得します。</para>
    /// </summary>
    public ReadOnlyMatrixSpan<double> Scale => new(_scale, _dimension, _dimension);

    /// <inheritdoc/>
    public Matrix<double> GetMean()
    {
        int p = _dimension;
        double denominator = _degreesOfFreedom - p - 1;
        var mean = Matrix<double>.Uninitialized(p, p);
        var data = mean.AsSpan();
        if (denominator > 0)
        {
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = _scale[i] / denominator;
            }

            return mean;
        }

        data.Fill(double.NaN);
        mean.Diagonal().Fill(double.PositiveInfinity);
        return mean;
    }

    /// <summary>
    /// Computes the elementwise variance matrix of the distribution.
    /// <para>分布の各成分の分散行列を計算します。</para>
    /// </summary>
    /// <returns>A newly allocated <c>p × p</c> matrix of elementwise variances: finite for <c>ν &gt; p + 3</c>, all <see cref="double.PositiveInfinity"/> for <c>p + 1 &lt; ν ≤ p + 3</c>, and all <see cref="double.NaN"/> for <c>ν ≤ p + 1</c>. / 各成分の分散を格納した新しい <c>p × p</c> 行列。<c>ν &gt; p + 3</c> で有限、<c>p + 1 &lt; ν ≤ p + 3</c> で全成分 <see cref="double.PositiveInfinity"/>、<c>ν ≤ p + 1</c> で全成分 <see cref="double.NaN"/>。</returns>
    public Matrix<double> GetVariance()
    {
        int p = _dimension;
        double nu = _degreesOfFreedom;
        var variance = Matrix<double>.Uninitialized(p, p);
        if (!(nu > p + 3))
        {
            variance.AsSpan().Fill(nu > p + 1 ? double.PositiveInfinity : double.NaN);
            return variance;
        }

        double m = nu - p - 1;
        double denominator = (nu - p) * m * m * (nu - p - 3);
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < p; i++)
            {
                double sij = _scale[i + j * p];
                variance[i, j] = ((nu - p + 1) * sij * sij + m * _scale[i + i * p] * _scale[j + j * p]) / denominator;
            }
        }

        return variance;
    }

    /// <inheritdoc/>
    public double Pdf(ReadOnlyMatrixSpan<double> x) => Math.Exp(LogPdf(x));

    /// <inheritdoc/>
    public double LogPdf(ReadOnlyMatrixSpan<double> x)
    {
        int p = _dimension;
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
        bool small = 2 * length <= MultivariateMath.StackLimit;
        Span<double> c = small ? stackalloc double[length] : new double[length];
        Span<double> b = small ? stackalloc double[length] : new double[length];
        MultivariateMath.CopyLowerSymmetric(x, c, p);
        if (!MultivariateMath.FactorInPlace(c, p))
        {
            return double.NegativeInfinity;
        }

        // B = C⁻¹L (lower triangular).
        _factor.CopyTo(b);
        for (int j = 0; j < p; j++)
        {
            MultivariateMath.SolveLowerInPlace(c, p, b, offset: j * p, start: j);
        }

        double sum = _logNormalization;
        for (int j = 0; j < p; j++)
        {
            double k = 0.5 * (_degreesOfFreedom - p + 1 + j);
            sum += MultivariateMath.LogGammaPrefixOfHalfSquare(k, b[j + j * p], out double logY) + (p - 0.5 * j) * logY;
            for (int i = j + 1; i < p; i++)
            {
                double value = b[i + j * p];
                sum -= 0.5 * value * value;
            }
        }

        // If tr(ΨX⁻¹) = ||B||² overflows for near-singular X, ∞ - ∞ yields NaN while the true density is 0.
        return double.IsNaN(sum) ? double.NegativeInfinity : sum;
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, MatrixSpan<double> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        int p = _dimension;
        MultivariateMath.RequireShape(destination, p, p, nameof(destination));
        int length = p * p;
        bool small = 2 * length <= MultivariateMath.StackLimit;
        Span<double> g = small ? stackalloc double[length] : new double[length];
        Span<double> t = small ? stackalloc double[length] : new double[length];
        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < i; j++)
            {
                g[i + j * p] = Ziggurat.StandardNormal(random);
            }

            g[i + i * p] = MultivariateMath.ChiSample(random, 0.5 * (_degreesOfFreedom - p + 1 + i));
        }

        // T = L G⁻¹ where G is the lower-triangular Bartlett factor.
        t.Clear();
        for (int j = 0; j < p; j++)
        {
            t[j + j * p] = 1;
            MultivariateMath.SolveLowerInPlace(g, p, t, offset: j * p, start: j);
        }

        MultivariateMath.MultiplyLowerInPlace(_factor, p, t);
        MultivariateMath.WriteLowerGram(t, p, destination);
    }
}
