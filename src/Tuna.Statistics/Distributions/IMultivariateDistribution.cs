using Tuna.LinearAlgebra;
using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents a continuous multivariate probability distribution over real vectors.
/// <para>実数ベクトルをとる多変量連続確率分布を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Distribution parameters are validated at construction time and copied defensively; instances are thread-safe for concurrent evaluation and sampling (using distinct <see cref="IRandomSource"/> instances per thread).
/// Each column of a <c>Dimension × N</c> matrix represents a single evaluation point or sample. Batch evaluation and batch sampling produce the same results and consume random values in the same order as column-by-column single-point calls.
/// </para>
/// <para>
/// 分布のパラメーターは構築時に検証され、配列などの入力値はコピーして保持されます。インスタンスは複数スレッドから安全に利用できます（乱数源 <see cref="IRandomSource"/> はスレッドごとに独立して使用してください）。
/// <c>Dimension × N</c> 行列の各列が 1 個の評価点または標本に対応します。一括評価および一括サンプリングは、列順に 1 点ずつ呼び出した場合と同じ結果と乱数消費順になります。
/// </para>
/// </remarks>
public interface IMultivariateContinuousDistribution
{
    /// <summary>
    /// Gets the dimension (length of the random vector).
    /// <para>次元（確率ベクトルの長さ）を取得します。</para>
    /// </summary>
    int Dimension { get; }

    /// <summary>
    /// Computes the mean vector of the distribution.
    /// <para>分布の平均ベクトルを計算します。</para>
    /// </summary>
    /// <returns>A newly allocated vector of length <see cref="Dimension"/> containing the mean; undefined entries are <see cref="double.NaN"/> and entries diverging to positive infinity are <see cref="double.PositiveInfinity"/>. / 平均を格納した長さ <see cref="Dimension"/> の新しいベクトル。定義されない成分は <see cref="double.NaN"/>、正に発散する成分は <see cref="double.PositiveInfinity"/>。</returns>
    Vector<double> GetMean();

    /// <summary>
    /// Computes the covariance matrix of the distribution.
    /// <para>分布の共分散行列を計算します。</para>
    /// </summary>
    /// <returns>A newly allocated <see cref="Dimension"/> × <see cref="Dimension"/> matrix containing the covariance; undefined entries are <see cref="double.NaN"/> and entries diverging to positive infinity are <see cref="double.PositiveInfinity"/>. / 共分散を格納した <see cref="Dimension"/> × <see cref="Dimension"/> の新しい行列。定義されない成分は <see cref="double.NaN"/>、正に発散する成分は <see cref="double.PositiveInfinity"/>。</returns>
    Matrix<double> GetCovariance();

    /// <summary>
    /// Evaluates the probability density function (PDF) at the specified vector.
    /// <para>指定したベクトルで確率密度関数（PDF）を評価します。</para>
    /// </summary>
    /// <param name="x">The evaluation vector of length <see cref="Dimension"/>. / 長さ <see cref="Dimension"/> の評価点ベクトル。</param>
    /// <returns>The probability density, 0 outside the support, or <see cref="double.NaN"/> if <paramref name="x"/> contains <see cref="double.NaN"/>. / 確率密度。台の外では 0、<paramref name="x"/> に <see cref="double.NaN"/> を含む場合は <see cref="double.NaN"/>。</returns>
    /// <exception cref="ArgumentException">The length of <paramref name="x"/> does not equal <see cref="Dimension"/>. / <paramref name="x"/> の長さが <see cref="Dimension"/> と一致しない場合。</exception>
    double Pdf(ReadOnlyVectorSpan<double> x);

    /// <summary>
    /// Evaluates the natural logarithm of the probability density function at the specified vector.
    /// <para>指定したベクトルで確率密度関数の自然対数を評価します。</para>
    /// </summary>
    /// <param name="x">The evaluation vector of length <see cref="Dimension"/>. / 長さ <see cref="Dimension"/> の評価点ベクトル。</param>
    /// <returns>The log-density, <see cref="double.NegativeInfinity"/> outside the support, or <see cref="double.NaN"/> if <paramref name="x"/> contains <see cref="double.NaN"/>. / 確率密度の自然対数。台の外では <see cref="double.NegativeInfinity"/>、<paramref name="x"/> に <see cref="double.NaN"/> を含む場合は <see cref="double.NaN"/>。</returns>
    /// <exception cref="ArgumentException">The length of <paramref name="x"/> does not equal <see cref="Dimension"/>. / <paramref name="x"/> の長さが <see cref="Dimension"/> と一致しない場合。</exception>
    double LogPdf(ReadOnlyVectorSpan<double> x);

    /// <summary>
    /// Evaluates the natural logarithm of the probability density function for each column of the input matrix.
    /// <para>入力行列の各列を評価点として、確率密度関数の自然対数を計算します。</para>
    /// </summary>
    /// <param name="x">The matrix of evaluation points with <see cref="Dimension"/> rows. / 行数が <see cref="Dimension"/> の評価点行列。</param>
    /// <param name="destination">The destination vector of length equal to the column count of <paramref name="x"/>, non-overlapping with <paramref name="x"/>. / 結果の書き込み先ベクトル。<paramref name="x"/> の列数と同じ長さで、<paramref name="x"/> とメモリ領域が重ならないこと。</param>
    /// <exception cref="ArgumentException">The shapes do not match, or <paramref name="destination"/> overlaps with <paramref name="x"/>. / 形状が一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    void LogPdf(ReadOnlyMatrixSpan<double> x, VectorSpan<double> destination);

    /// <summary>
    /// Draws a single random vector from the distribution and writes it to the destination.
    /// <para>分布に従って標本ベクトルを 1 個生成し、出力先へ書き込みます。</para>
    /// </summary>
    /// <param name="random">The uniform random number source. / 一様乱数源。</param>
    /// <param name="destination">The destination vector of length <see cref="Dimension"/>. / 長さ <see cref="Dimension"/> の書き込み先ベクトル。</param>
    /// <exception cref="ArgumentNullException"><paramref name="random"/> is <see langword="null"/>. / <paramref name="random"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <see cref="Dimension"/>. / <paramref name="destination"/> の長さが <see cref="Dimension"/> と一致しない場合。</exception>
    void Sample(IRandomSource random, VectorSpan<double> destination);

    /// <summary>
    /// Draws random vectors from the distribution and writes one sample per column of the destination matrix.
    /// <para>分布に従って標本ベクトルを生成し、出力先行列の各列に 1 個ずつ書き込みます。</para>
    /// </summary>
    /// <param name="random">The uniform random number source. / 一様乱数源。</param>
    /// <param name="destination">The destination matrix with <see cref="Dimension"/> rows; one sample is written per column. / 行数が <see cref="Dimension"/> の書き込み先行列。列数と同数の標本を各列に書き込みます。</param>
    /// <exception cref="ArgumentNullException"><paramref name="random"/> is <see langword="null"/>. / <paramref name="random"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException">The row count of <paramref name="destination"/> does not equal <see cref="Dimension"/>. / <paramref name="destination"/> の行数が <see cref="Dimension"/> と一致しない場合。</exception>
    void Sample(IRandomSource random, MatrixSpan<double> destination);
}

/// <summary>
/// Represents a discrete multivariate probability distribution over integer vectors.
/// <para>整数ベクトルをとる多変量離散確率分布を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Distribution parameters are validated at construction time and copied defensively; instances are thread-safe for concurrent evaluation and sampling (using distinct <see cref="IRandomSource"/> instances per thread).
/// Each column of a <c>Dimension × N</c> matrix represents a single evaluation point or sample. Batch evaluation and batch sampling produce the same results and consume random values in the same order as column-by-column single-point calls.
/// </para>
/// <para>
/// 分布のパラメーターは構築時に検証され、配列などの入力値はコピーして保持されます。インスタンスは複数スレッドから安全に利用できます（乱数源 <see cref="IRandomSource"/> はスレッドごとに独立して使用してください）。
/// <c>Dimension × N</c> 行列の各列が 1 個の評価点または標本に対応します。一括評価および一括サンプリングは、列順に 1 点ずつ呼び出した場合と同じ結果と乱数消費順になります。
/// </para>
/// </remarks>
public interface IMultivariateDiscreteDistribution
{
    /// <summary>
    /// Gets the dimension (length of the random vector).
    /// <para>次元（確率ベクトルの長さ）を取得します。</para>
    /// </summary>
    int Dimension { get; }

    /// <summary>
    /// Computes the mean vector of the distribution.
    /// <para>分布の平均ベクトルを計算します。</para>
    /// </summary>
    /// <returns>A newly allocated vector of length <see cref="Dimension"/> containing the mean; undefined entries are <see cref="double.NaN"/> and entries diverging to positive infinity are <see cref="double.PositiveInfinity"/>. / 平均を格納した長さ <see cref="Dimension"/> の新しいベクトル。定義されない成分は <see cref="double.NaN"/>、正に発散する成分は <see cref="double.PositiveInfinity"/>。</returns>
    Vector<double> GetMean();

    /// <summary>
    /// Computes the covariance matrix of the distribution.
    /// <para>分布の共分散行列を計算します。</para>
    /// </summary>
    /// <returns>A newly allocated <see cref="Dimension"/> × <see cref="Dimension"/> matrix containing the covariance; undefined entries are <see cref="double.NaN"/> and entries diverging to positive infinity are <see cref="double.PositiveInfinity"/>. / 共分散を格納した <see cref="Dimension"/> × <see cref="Dimension"/> の新しい行列。定義されない成分は <see cref="double.NaN"/>、正に発散する成分は <see cref="double.PositiveInfinity"/>。</returns>
    Matrix<double> GetCovariance();

    /// <summary>
    /// Evaluates the probability mass function (PMF) at the specified integer vector.
    /// <para>指定した整数ベクトルで確率質量関数（PMF）を評価します。</para>
    /// </summary>
    /// <param name="x">The evaluation vector of length <see cref="Dimension"/>. / 長さ <see cref="Dimension"/> の評価点ベクトル。</param>
    /// <returns>The probability mass, or 0 outside the support. / 確率質量。台の外では 0。</returns>
    /// <exception cref="ArgumentException">The length of <paramref name="x"/> does not equal <see cref="Dimension"/>. / <paramref name="x"/> の長さが <see cref="Dimension"/> と一致しない場合。</exception>
    double Pmf(ReadOnlyVectorSpan<long> x);

    /// <summary>
    /// Evaluates the natural logarithm of the probability mass function at the specified integer vector.
    /// <para>指定した整数ベクトルで確率質量関数の自然対数を評価します。</para>
    /// </summary>
    /// <param name="x">The evaluation vector of length <see cref="Dimension"/>. / 長さ <see cref="Dimension"/> の評価点ベクトル。</param>
    /// <returns>The log-probability mass, or <see cref="double.NegativeInfinity"/> outside the support. / 確率質量の自然対数。台の外では <see cref="double.NegativeInfinity"/>。</returns>
    /// <exception cref="ArgumentException">The length of <paramref name="x"/> does not equal <see cref="Dimension"/>. / <paramref name="x"/> の長さが <see cref="Dimension"/> と一致しない場合。</exception>
    double LogPmf(ReadOnlyVectorSpan<long> x);

    /// <summary>
    /// Evaluates the natural logarithm of the probability mass function for each column of the input matrix.
    /// <para>入力行列の各列を評価点として、確率質量関数の自然対数を計算します。</para>
    /// </summary>
    /// <param name="x">The matrix of evaluation points with <see cref="Dimension"/> rows. / 行数が <see cref="Dimension"/> の評価点行列。</param>
    /// <param name="destination">The destination vector of length equal to the column count of <paramref name="x"/>, non-overlapping with <paramref name="x"/>. / 結果の書き込み先ベクトル。<paramref name="x"/> の列数と同じ長さで、<paramref name="x"/> とメモリ領域が重ならないこと。</param>
    /// <exception cref="ArgumentException">The shapes do not match, or <paramref name="destination"/> overlaps with <paramref name="x"/>. / 形状が一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    void LogPmf(ReadOnlyMatrixSpan<long> x, VectorSpan<double> destination);

    /// <summary>
    /// Draws a single random integer vector from the distribution and writes it to the destination.
    /// <para>分布に従って整数標本ベクトルを 1 個生成し、出力先へ書き込みます。</para>
    /// </summary>
    /// <param name="random">The uniform random number source. / 一様乱数源。</param>
    /// <param name="destination">The destination vector of length <see cref="Dimension"/>. / 長さ <see cref="Dimension"/> の書き込み先ベクトル。</param>
    /// <exception cref="ArgumentNullException"><paramref name="random"/> is <see langword="null"/>. / <paramref name="random"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <see cref="Dimension"/>. / <paramref name="destination"/> の長さが <see cref="Dimension"/> と一致しない場合。</exception>
    void Sample(IRandomSource random, VectorSpan<long> destination);

    /// <summary>
    /// Draws random integer vectors from the distribution and writes one sample per column of the destination matrix.
    /// <para>分布に従って整数標本ベクトルを生成し、出力先行列の各列に 1 個ずつ書き込みます。</para>
    /// </summary>
    /// <param name="random">The uniform random number source. / 一様乱数源。</param>
    /// <param name="destination">The destination matrix with <see cref="Dimension"/> rows; one sample is written per column. / 行数が <see cref="Dimension"/> の書き込み先行列。列数と同数の標本を各列に書き込みます。</param>
    /// <exception cref="ArgumentNullException"><paramref name="random"/> is <see langword="null"/>. / <paramref name="random"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException">The row count of <paramref name="destination"/> does not equal <see cref="Dimension"/>. / <paramref name="destination"/> の行数が <see cref="Dimension"/> と一致しない場合。</exception>
    void Sample(IRandomSource random, MatrixSpan<long> destination);
}

/// <summary>
/// Represents a continuous probability distribution over matrices.
/// <para>行列値をとる連続確率分布を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// For distributions over symmetric matrices, the density is with respect to Lebesgue measure on the independent lower-triangular entries (excluding the diagonal for the LKJ correlation distribution).
/// Matrices that are asymmetric beyond floating-point rounding tolerance are treated as outside the support.
/// </para>
/// <para>
/// 対称行列の分布における密度は、独立な下三角成分に対するルベーグ測度に関して定義されます（LKJ 相関分布では対角成分を除きます）。
/// 丸め誤差の許容範囲を超えて非対称な行列は台の外として扱われます。
/// </para>
/// </remarks>
public interface IMatrixContinuousDistribution
{
    /// <summary>
    /// Gets the number of rows of the random matrix.
    /// <para>確率行列の行数を取得します。</para>
    /// </summary>
    int Rows { get; }

    /// <summary>
    /// Gets the number of columns of the random matrix.
    /// <para>確率行列の列数を取得します。</para>
    /// </summary>
    int Columns { get; }

    /// <summary>
    /// Computes the mean matrix of the distribution.
    /// <para>分布の平均行列を計算します。</para>
    /// </summary>
    /// <returns>A newly allocated <see cref="Rows"/> × <see cref="Columns"/> matrix containing the mean; undefined entries are <see cref="double.NaN"/> and entries diverging to positive infinity are <see cref="double.PositiveInfinity"/>. / 平均を格納した <see cref="Rows"/> × <see cref="Columns"/> の新しい行列。定義されない成分は <see cref="double.NaN"/>、正に発散する成分は <see cref="double.PositiveInfinity"/>。</returns>
    Matrix<double> GetMean();

    /// <summary>
    /// Evaluates the probability density function (PDF) at the specified matrix.
    /// <para>指定した行列で確率密度関数（PDF）を評価します。</para>
    /// </summary>
    /// <param name="x">The <see cref="Rows"/> × <see cref="Columns"/> evaluation matrix. / <see cref="Rows"/> × <see cref="Columns"/> の評価点行列。</param>
    /// <returns>The probability density, 0 outside the support, or <see cref="double.NaN"/> if <paramref name="x"/> contains <see cref="double.NaN"/>. / 確率密度。台の外では 0、<paramref name="x"/> に <see cref="double.NaN"/> を含む場合は <see cref="double.NaN"/>。</returns>
    /// <exception cref="ArgumentException">The shape of <paramref name="x"/> does not match <see cref="Rows"/> × <see cref="Columns"/>. / <paramref name="x"/> の形状が <see cref="Rows"/> × <see cref="Columns"/> と一致しない場合。</exception>
    double Pdf(ReadOnlyMatrixSpan<double> x);

    /// <summary>
    /// Evaluates the natural logarithm of the probability density function at the specified matrix.
    /// <para>指定した行列で確率密度関数の自然対数を評価します。</para>
    /// </summary>
    /// <param name="x">The <see cref="Rows"/> × <see cref="Columns"/> evaluation matrix. / <see cref="Rows"/> × <see cref="Columns"/> の評価点行列。</param>
    /// <returns>The log-density, <see cref="double.NegativeInfinity"/> outside the support, or <see cref="double.NaN"/> if <paramref name="x"/> contains <see cref="double.NaN"/>. / 確率密度の自然対数。台の外では <see cref="double.NegativeInfinity"/>、<paramref name="x"/> に <see cref="double.NaN"/> を含む場合は <see cref="double.NaN"/>。</returns>
    /// <exception cref="ArgumentException">The shape of <paramref name="x"/> does not match <see cref="Rows"/> × <see cref="Columns"/>. / <paramref name="x"/> の形状が <see cref="Rows"/> × <see cref="Columns"/> と一致しない場合。</exception>
    double LogPdf(ReadOnlyMatrixSpan<double> x);

    /// <summary>
    /// Draws a single random matrix from the distribution and writes it to the destination.
    /// <para>分布に従って標本行列を 1 個生成し、出力先へ書き込みます。</para>
    /// </summary>
    /// <param name="random">The uniform random number source. / 一様乱数源。</param>
    /// <param name="destination">The <see cref="Rows"/> × <see cref="Columns"/> destination matrix. / <see cref="Rows"/> × <see cref="Columns"/> の書き込み先行列。</param>
    /// <exception cref="ArgumentNullException"><paramref name="random"/> is <see langword="null"/>. / <paramref name="random"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException">The shape of <paramref name="destination"/> does not match <see cref="Rows"/> × <see cref="Columns"/>. / <paramref name="destination"/> の形状が <see cref="Rows"/> × <see cref="Columns"/> と一致しない場合。</exception>
    void Sample(IRandomSource random, MatrixSpan<double> destination);
}

/// <summary>
/// Provides allocating convenience overloads for multivariate and matrix distributions.
/// <para>多変量分布および行列値分布に対して、結果を新規確保して返す拡張メソッドを提供します。</para>
/// </summary>
public static class MultivariateDistributionExtensions
{
    extension(IMultivariateContinuousDistribution distribution)
    {
        /// <summary>
        /// Draws a single random vector from the distribution.
        /// <para>分布に従って標本ベクトルを 1 個生成します。</para>
        /// </summary>
        /// <param name="random">The uniform random number source. / 一様乱数源。</param>
        /// <returns>A newly allocated vector containing the drawn sample. / 生成した標本を格納した新しいベクトル。</returns>
        /// <exception cref="ArgumentNullException">The distribution or <paramref name="random"/> is <see langword="null"/>. / 分布または <paramref name="random"/> が <see langword="null"/> の場合。</exception>
        public Vector<double> Sample(IRandomSource random)
        {
            ArgumentNullException.ThrowIfNull(distribution);
            var result = Vector<double>.Uninitialized(distribution.Dimension);
            distribution.Sample(random, result);
            return result;
        }

        /// <summary>
        /// Draws multiple random vectors from the distribution.
        /// <para>分布に従って複数の標本ベクトルを生成します。</para>
        /// </summary>
        /// <param name="random">The uniform random number source. / 一様乱数源。</param>
        /// <param name="count">The number of samples to draw (non-negative). / 生成する標本数（0 以上）。</param>
        /// <returns>A newly allocated matrix with <c>Dimension</c> rows and <paramref name="count"/> columns, containing one sample per column. / 行数が <c>Dimension</c>、列数が <paramref name="count"/> の新しい行列。各列に 1 個の標本を格納します。</returns>
        /// <exception cref="ArgumentNullException">The distribution or <paramref name="random"/> is <see langword="null"/>. / 分布または <paramref name="random"/> が <see langword="null"/> の場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is negative. / <paramref name="count"/> が負の場合。</exception>
        /// <exception cref="OverflowException">The total element count exceeds <see cref="int.MaxValue"/>. / 総要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
        public Matrix<double> Sample(IRandomSource random, int count)
        {
            ArgumentNullException.ThrowIfNull(distribution);
            ArgumentOutOfRangeException.ThrowIfNegative(count);
            var result = Matrix<double>.Uninitialized(distribution.Dimension, count);
            distribution.Sample(random, result);
            return result;
        }

        /// <summary>
        /// Evaluates the natural logarithm of the probability density function for each column of the input matrix.
        /// <para>入力行列の各列を評価点として、確率密度関数の自然対数を計算します。</para>
        /// </summary>
        /// <param name="x">The matrix of evaluation points with <c>Dimension</c> rows. / 行数が <c>Dimension</c> の評価点行列。</param>
        /// <returns>A newly allocated vector containing the log-density for each column in order (<see cref="double.NegativeInfinity"/> outside the support, <see cref="double.NaN"/> if a column contains <see cref="double.NaN"/>). / 各列の対数密度を入力の列順に格納した新しいベクトル。台の外では <see cref="double.NegativeInfinity"/>、<see cref="double.NaN"/> を含む列では <see cref="double.NaN"/>。</returns>
        /// <exception cref="ArgumentNullException">The distribution is <see langword="null"/>. / 分布が <see langword="null"/> の場合。</exception>
        /// <exception cref="ArgumentException">The row count of <paramref name="x"/> does not equal <c>Dimension</c>. / <paramref name="x"/> の行数が <c>Dimension</c> と一致しない場合。</exception>
        public Vector<double> LogPdf(ReadOnlyMatrixSpan<double> x)
        {
            ArgumentNullException.ThrowIfNull(distribution);
            var result = Vector<double>.Uninitialized(x.Columns);
            distribution.LogPdf(x, result);
            return result;
        }
    }

    extension(IMultivariateDiscreteDistribution distribution)
    {
        /// <summary>
        /// Draws a single random integer vector from the distribution.
        /// <para>分布に従って整数標本ベクトルを 1 個生成します。</para>
        /// </summary>
        /// <param name="random">The uniform random number source. / 一様乱数源。</param>
        /// <returns>A newly allocated vector containing the drawn sample. / 生成した標本を格納した新しいベクトル。</returns>
        /// <exception cref="ArgumentNullException">The distribution or <paramref name="random"/> is <see langword="null"/>. / 分布または <paramref name="random"/> が <see langword="null"/> の場合。</exception>
        public Vector<long> Sample(IRandomSource random)
        {
            ArgumentNullException.ThrowIfNull(distribution);
            var result = Vector<long>.Uninitialized(distribution.Dimension);
            distribution.Sample(random, result);
            return result;
        }

        /// <summary>
        /// Draws multiple random integer vectors from the distribution.
        /// <para>分布に従って複数の整数標本ベクトルを生成します。</para>
        /// </summary>
        /// <param name="random">The uniform random number source. / 一様乱数源。</param>
        /// <param name="count">The number of samples to draw (non-negative). / 生成する標本数（0 以上）。</param>
        /// <returns>A newly allocated matrix with <c>Dimension</c> rows and <paramref name="count"/> columns, containing one sample per column. / 行数が <c>Dimension</c>、列数が <paramref name="count"/> の新しい行列。各列に 1 個の標本を格納します。</returns>
        /// <exception cref="ArgumentNullException">The distribution or <paramref name="random"/> is <see langword="null"/>. / 分布または <paramref name="random"/> が <see langword="null"/> の場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is negative. / <paramref name="count"/> が負の場合。</exception>
        /// <exception cref="OverflowException">The total element count exceeds <see cref="int.MaxValue"/>. / 総要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
        public Matrix<long> Sample(IRandomSource random, int count)
        {
            ArgumentNullException.ThrowIfNull(distribution);
            ArgumentOutOfRangeException.ThrowIfNegative(count);
            var result = Matrix<long>.Uninitialized(distribution.Dimension, count);
            distribution.Sample(random, result);
            return result;
        }

        /// <summary>
        /// Evaluates the natural logarithm of the probability mass function for each column of the input matrix.
        /// <para>入力行列の各列を評価点として、確率質量関数の自然対数を計算します。</para>
        /// </summary>
        /// <param name="x">The matrix of evaluation points with <c>Dimension</c> rows. / 行数が <c>Dimension</c> の評価点行列。</param>
        /// <returns>A newly allocated vector containing the log-probability mass for each column in order (<see cref="double.NegativeInfinity"/> outside the support). / 各列の対数確率質量を入力の列順に格納した新しいベクトル。台の外では <see cref="double.NegativeInfinity"/>。</returns>
        /// <exception cref="ArgumentNullException">The distribution is <see langword="null"/>. / 分布が <see langword="null"/> の場合。</exception>
        /// <exception cref="ArgumentException">The row count of <paramref name="x"/> does not equal <c>Dimension</c>. / <paramref name="x"/> の行数が <c>Dimension</c> と一致しない場合。</exception>
        public Vector<double> LogPmf(ReadOnlyMatrixSpan<long> x)
        {
            ArgumentNullException.ThrowIfNull(distribution);
            var result = Vector<double>.Uninitialized(x.Columns);
            distribution.LogPmf(x, result);
            return result;
        }
    }

    extension(IMatrixContinuousDistribution distribution)
    {
        /// <summary>
        /// Draws a single random matrix from the distribution.
        /// <para>分布に従って標本行列を 1 個生成します。</para>
        /// </summary>
        /// <param name="random">The uniform random number source. / 一様乱数源。</param>
        /// <returns>A newly allocated matrix containing the drawn sample. / 生成した標本を格納した新しい行列。</returns>
        /// <exception cref="ArgumentNullException">The distribution or <paramref name="random"/> is <see langword="null"/>. / 分布または <paramref name="random"/> が <see langword="null"/> の場合。</exception>
        public Matrix<double> Sample(IRandomSource random)
        {
            ArgumentNullException.ThrowIfNull(distribution);
            var result = Matrix<double>.Uninitialized(distribution.Rows, distribution.Columns);
            distribution.Sample(random, result);
            return result;
        }
    }
}
