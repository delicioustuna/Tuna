using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents a univariate probability distribution.
/// <para>1 変量の確率分布を表します。</para>
/// </summary>
/// <typeparam name="TValue">The value type of the random variable (<see cref="double"/> for continuous distributions, <see cref="long"/> for discrete distributions). / 確率変数の型（連続分布では <see cref="double"/>、離散分布では <see cref="long"/>）。</typeparam>
/// <remarks>
/// <para>
/// Distribution parameters are validated at construction time.
/// For continuous distributions, evaluating probability functions at <see cref="double.NaN"/> returns <see cref="double.NaN"/>.
/// For out-of-range inputs to inverse functions, see <see cref="IContinuousDistribution"/> and <see cref="IDiscreteDistribution"/>.
/// </para>
/// <para>
/// 分布のパラメーターは構築時に検証されます。
/// 連続分布の確率計算で評価点に <see cref="double.NaN"/> を指定した場合は <see cref="double.NaN"/> を返します。
/// 逆関数に範囲外の確率を指定した場合の振る舞いは <see cref="IContinuousDistribution"/> および <see cref="IDiscreteDistribution"/> を参照してください。
/// </para>
/// </remarks>
public interface IUnivariateDistribution<TValue>
{
    /// <summary>
    /// Gets the mean (expected value) of the distribution.
    /// <para>分布の平均（期待値）を取得します。</para>
    /// </summary>
    /// <value>The mean, <see cref="double.NaN"/> if undefined, or <see cref="double.PositiveInfinity"/> if it diverges to positive infinity. / 平均。定義されない場合は <see cref="double.NaN"/>、正に発散する場合は <see cref="double.PositiveInfinity"/>。</value>
    double Mean { get; }

    /// <summary>
    /// Gets the variance of the distribution.
    /// <para>分布の分散を取得します。</para>
    /// </summary>
    /// <value>The variance, <see cref="double.NaN"/> if undefined, or <see cref="double.PositiveInfinity"/> if it diverges to positive infinity. / 分散。定義されない場合は <see cref="double.NaN"/>、正に発散する場合は <see cref="double.PositiveInfinity"/>。</value>
    double Variance { get; }

    /// <summary>
    /// Gets the standard deviation of the distribution.
    /// <para>分布の標準偏差を取得します。</para>
    /// </summary>
    /// <value>The standard deviation, <see cref="double.NaN"/> if undefined, or <see cref="double.PositiveInfinity"/> if it diverges to positive infinity. / 標準偏差。定義されない場合は <see cref="double.NaN"/>、正に発散する場合は <see cref="double.PositiveInfinity"/>。</value>
    double StandardDeviation { get; }

    /// <summary>
    /// Gets the skewness of the distribution.
    /// <para>分布の歪度を取得します。</para>
    /// </summary>
    /// <value>The skewness, or <see cref="double.NaN"/> if undefined. / 歪度。定義されない場合は <see cref="double.NaN"/>。</value>
    double Skewness { get; }

    /// <summary>
    /// Gets the excess kurtosis of the distribution.
    /// <para>分布の超過尖度を取得します。</para>
    /// </summary>
    /// <value>The excess kurtosis (0 for a normal distribution), or <see cref="double.NaN"/> if undefined. / 超過尖度（正規分布では 0）。定義されない場合は <see cref="double.NaN"/>。</value>
    double ExcessKurtosis { get; }

    /// <summary>
    /// Gets the entropy of the distribution in nats.
    /// <para>分布のエントロピー（単位: nat）を取得します。</para>
    /// </summary>
    /// <value>The natural-logarithm entropy (differential entropy for continuous distributions), in nats. / 自然対数によるエントロピー（連続分布では微分エントロピー、単位は nat）。</value>
    double Entropy { get; }

    /// <summary>
    /// Gets the median of the distribution.
    /// <para>分布の中央値を取得します。</para>
    /// </summary>
    /// <value>The median, equal to <c>InverseCdf(0.5)</c>. / 中央値（<c>InverseCdf(0.5)</c> と同じ値）。</value>
    TValue Median { get; }

    /// <summary>
    /// Gets the lower bound of the distribution's support.
    /// <para>分布の台の下限を取得します。</para>
    /// </summary>
    /// <value>The lower bound of the support, or <see cref="double.NegativeInfinity"/> / <see cref="long.MinValue"/> if unbounded below. / 台の下限。下に有界でない場合は <see cref="double.NegativeInfinity"/> または <see cref="long.MinValue"/>。</value>
    TValue SupportMinimum { get; }

    /// <summary>
    /// Gets the upper bound of the distribution's support.
    /// <para>分布の台の上限を取得します。</para>
    /// </summary>
    /// <value>The upper bound of the support, or <see cref="double.PositiveInfinity"/> / <see cref="long.MaxValue"/> if unbounded above. / 台の上限。上に有界でない場合は <see cref="double.PositiveInfinity"/> または <see cref="long.MaxValue"/>。</value>
    TValue SupportMaximum { get; }

    /// <summary>
    /// Evaluates the cumulative distribution function (CDF), <c>P(X ≤ x)</c>.
    /// <para>累積分布関数（CDF）<c>P(X ≤ x)</c> を評価します。</para>
    /// </summary>
    /// <param name="x">The evaluation point. / 評価点。</param>
    /// <returns><c>P(X ≤ x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is <see cref="double.NaN"/> for a continuous distribution. / <c>P(X ≤ x)</c>。連続分布で <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
    double Cdf(TValue x);

    /// <summary>
    /// Evaluates the survival function (complementary CDF), <c>P(X &gt; x)</c>.
    /// <para>生存関数（上側累積確率）<c>P(X &gt; x)</c> を評価します。</para>
    /// </summary>
    /// <param name="x">The evaluation point. / 評価点。</param>
    /// <returns><c>P(X &gt; x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is <see cref="double.NaN"/> for a continuous distribution. / <c>P(X &gt; x)</c>。連続分布で <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
    double Sf(TValue x);

    /// <summary>
    /// Evaluates the natural logarithm of the cumulative distribution function, <c>ln P(X ≤ x)</c>.
    /// <para>累積分布関数の自然対数 <c>ln P(X ≤ x)</c> を評価します。</para>
    /// </summary>
    /// <param name="x">The evaluation point. / 評価点。</param>
    /// <returns><c>ln P(X ≤ x)</c>, <see cref="double.NegativeInfinity"/> when the probability is 0, or <see cref="double.NaN"/> if <paramref name="x"/> is <see cref="double.NaN"/> for a continuous distribution. / <c>ln P(X ≤ x)</c>。確率が 0 の場合は <see cref="double.NegativeInfinity"/>、連続分布で <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
    double LogCdf(TValue x);

    /// <summary>
    /// Evaluates the natural logarithm of the survival function, <c>ln P(X &gt; x)</c>.
    /// <para>生存関数の自然対数 <c>ln P(X &gt; x)</c> を評価します。</para>
    /// </summary>
    /// <param name="x">The evaluation point. / 評価点。</param>
    /// <returns><c>ln P(X &gt; x)</c>, <see cref="double.NegativeInfinity"/> when the probability is 0, or <see cref="double.NaN"/> if <paramref name="x"/> is <see cref="double.NaN"/> for a continuous distribution. / <c>ln P(X &gt; x)</c>。確率が 0 の場合は <see cref="double.NegativeInfinity"/>、連続分布で <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
    double LogSf(TValue x);

    /// <summary>
    /// Computes the quantile corresponding to the specified cumulative probability.
    /// <para>指定した累積確率に対応する分位点を計算します。</para>
    /// </summary>
    /// <param name="p">The cumulative probability in <c>[0, 1]</c>. / <c>[0, 1]</c> の範囲の累積確率。</param>
    /// <returns>The quantile satisfying <c>Cdf(x) ≥ p</c> (the smallest such integer for discrete distributions). Returns <see cref="SupportMinimum"/> when <paramref name="p"/> is 0 and <see cref="SupportMaximum"/> when <paramref name="p"/> is 1. / <c>Cdf(x) ≥ p</c> を満たす分位点（離散分布では最小の整数）。<paramref name="p"/> が 0 の場合は <see cref="SupportMinimum"/>、1 の場合は <see cref="SupportMaximum"/>。</returns>
    /// <exception cref="ArgumentOutOfRangeException">For discrete distributions, <paramref name="p"/> is <see cref="double.NaN"/> or outside <c>[0, 1]</c>. / 離散分布で <paramref name="p"/> が <see cref="double.NaN"/> または <c>[0, 1]</c> の範囲外の場合。</exception>
    TValue InverseCdf(double p);

    /// <summary>
    /// Computes the quantile corresponding to the specified upper-tail probability.
    /// <para>指定した上側確率に対応する分位点を計算します。</para>
    /// </summary>
    /// <param name="q">The upper-tail probability in <c>[0, 1]</c>. / <c>[0, 1]</c> の範囲の上側確率。</param>
    /// <returns>The quantile satisfying <c>Sf(x) ≤ q</c> (the smallest such integer for discrete distributions). Returns <see cref="SupportMinimum"/> when <paramref name="q"/> is 1 and <see cref="SupportMaximum"/> when <paramref name="q"/> is 0. / <c>Sf(x) ≤ q</c> を満たす分位点（離散分布では最小の整数）。<paramref name="q"/> が 1 の場合は <see cref="SupportMinimum"/>、0 の場合は <see cref="SupportMaximum"/>。</returns>
    /// <exception cref="ArgumentOutOfRangeException">For discrete distributions, <paramref name="q"/> is <see cref="double.NaN"/> or outside <c>[0, 1]</c>. / 離散分布で <paramref name="q"/> が <see cref="double.NaN"/> または <c>[0, 1]</c> の範囲外の場合。</exception>
    TValue InverseSf(double q);

    /// <summary>
    /// Computes the quantile corresponding to the specified log cumulative probability.
    /// <para>指定した対数累積確率に対応する分位点を計算します。</para>
    /// </summary>
    /// <param name="logP">The natural logarithm of the cumulative probability (<c>≤ 0</c>). / 累積確率の自然対数（0 以下）。</param>
    /// <returns>The quantile satisfying <c>LogCdf(x) ≥ logP</c> (the smallest such integer for discrete distributions). Returns <see cref="SupportMinimum"/> when <paramref name="logP"/> is <see cref="double.NegativeInfinity"/> and <see cref="SupportMaximum"/> when <paramref name="logP"/> is 0. / <c>LogCdf(x) ≥ logP</c> を満たす分位点（離散分布では最小の整数）。<paramref name="logP"/> が <see cref="double.NegativeInfinity"/> の場合は <see cref="SupportMinimum"/>、0 の場合は <see cref="SupportMaximum"/>。</returns>
    /// <exception cref="ArgumentOutOfRangeException">For discrete distributions, <paramref name="logP"/> is <see cref="double.NaN"/> or positive. / 離散分布で <paramref name="logP"/> が <see cref="double.NaN"/> または正の場合。</exception>
    TValue InverseLogCdf(double logP);

    /// <summary>
    /// Computes the quantile corresponding to the specified log upper-tail probability.
    /// <para>指定した対数上側確率に対応する分位点を計算します。</para>
    /// </summary>
    /// <param name="logQ">The natural logarithm of the upper-tail probability (<c>≤ 0</c>). / 上側確率の自然対数（0 以下）。</param>
    /// <returns>The quantile satisfying <c>LogSf(x) ≤ logQ</c> (the smallest such integer for discrete distributions). Returns <see cref="SupportMinimum"/> when <paramref name="logQ"/> is 0 and <see cref="SupportMaximum"/> when <paramref name="logQ"/> is <see cref="double.NegativeInfinity"/>. / <c>LogSf(x) ≤ logQ</c> を満たす分位点（離散分布では最小の整数）。<paramref name="logQ"/> が 0 の場合は <see cref="SupportMinimum"/>、<see cref="double.NegativeInfinity"/> の場合は <see cref="SupportMaximum"/>。</returns>
    /// <exception cref="ArgumentOutOfRangeException">For discrete distributions, <paramref name="logQ"/> is <see cref="double.NaN"/> or positive. / 離散分布で <paramref name="logQ"/> が <see cref="double.NaN"/> または正の場合。</exception>
    TValue InverseLogSf(double logQ);

    /// <summary>
    /// Draws a single random sample from the distribution.
    /// <para>分布に従って標本を 1 個生成します。</para>
    /// </summary>
    /// <param name="random">The uniform random number source. / 一様乱数源。</param>
    /// <returns>A single sample drawn from the distribution. / 分布に従って生成した 1 個の標本。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="random"/> is <see langword="null"/>. / <paramref name="random"/> が <see langword="null"/> の場合。</exception>
    TValue Sample(IRandomSource random);

    /// <summary>
    /// Fills a span with random samples drawn from the distribution.
    /// <para>分布に従って生成した標本をスパンに書き込みます。</para>
    /// </summary>
    /// <param name="random">The uniform random number source. / 一様乱数源。</param>
    /// <param name="destination">The span to fill with samples. / 標本を書き込むスパン。</param>
    /// <remarks>
    /// <para>
    /// Produces the same samples and consumes random values in the same order as calling <see cref="Sample(IRandomSource)"/> sequentially for each element.
    /// </para>
    /// <para>
    /// 各要素に対して <see cref="Sample(IRandomSource)"/> を順に呼び出した場合と、生成される標本および乱数の消費順が一致します。
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="random"/> is <see langword="null"/>. / <paramref name="random"/> が <see langword="null"/> の場合。</exception>
    void Sample(IRandomSource random, Span<TValue> destination);
}

/// <summary>
/// Represents a continuous univariate probability distribution.
/// <para>連続値をとる 1 変量の確率分布を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Inverse functions return <see cref="double.NaN"/> when given <see cref="double.NaN"/> or an out-of-range input (a probability outside <c>[0, 1]</c> or a positive log-probability).
/// </para>
/// <para>
/// 逆関数の引数が <see cref="double.NaN"/> または範囲外（確率が <c>[0, 1]</c> の外、または対数確率が正）の場合は <see cref="double.NaN"/> を返します。
/// </para>
/// </remarks>
public interface IContinuousDistribution : IUnivariateDistribution<double>
{
    /// <summary>
    /// Evaluates the probability density function (PDF) at the specified point.
    /// <para>指定した点で確率密度関数（PDF）を評価します。</para>
    /// </summary>
    /// <param name="x">The evaluation point. / 評価点。</param>
    /// <returns>The probability density, 0 outside the support, or <see cref="double.NaN"/> if <paramref name="x"/> is <see cref="double.NaN"/>. / 確率密度。台の外では 0、<paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
    double Pdf(double x);

    /// <summary>
    /// Evaluates the natural logarithm of the probability density function at the specified point.
    /// <para>指定した点で確率密度関数の自然対数を評価します。</para>
    /// </summary>
    /// <param name="x">The evaluation point. / 評価点。</param>
    /// <returns>The log-density, <see cref="double.NegativeInfinity"/> outside the support, or <see cref="double.NaN"/> if <paramref name="x"/> is <see cref="double.NaN"/>. / 確率密度の自然対数。台の外では <see cref="double.NegativeInfinity"/>、<paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
    double LogPdf(double x);
}

/// <summary>
/// Represents a discrete univariate probability distribution over integers.
/// <para>整数値をとる 1 変量の離散確率分布を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Inverse functions throw <see cref="ArgumentOutOfRangeException"/> when given <see cref="double.NaN"/> or an out-of-range input (a probability outside <c>[0, 1]</c> or a positive log-probability).
/// For distributions unbounded above, <see cref="IUnivariateDistribution{TValue}.SupportMaximum"/> is <see cref="long.MaxValue"/>.
/// </para>
/// <para>
/// 逆関数の引数が <see cref="double.NaN"/> または範囲外（確率が <c>[0, 1]</c> の外、または対数確率が正）の場合は <see cref="ArgumentOutOfRangeException"/> をスローします。
/// 上に有界でない分布の <see cref="IUnivariateDistribution{TValue}.SupportMaximum"/> は <see cref="long.MaxValue"/> です。
/// </para>
/// </remarks>
public interface IDiscreteDistribution : IUnivariateDistribution<long>
{
    /// <summary>
    /// Evaluates the probability mass function (PMF) at the specified integer.
    /// <para>指定した整数点で確率質量関数（PMF）を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns>The probability mass, or 0 outside the support. / 確率質量。台の外では 0。</returns>
    double Pmf(long k);

    /// <summary>
    /// Evaluates the natural logarithm of the probability mass function at the specified integer.
    /// <para>指定した整数点で確率質量関数の自然対数を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns>The log-probability mass, or <see cref="double.NegativeInfinity"/> outside the support. / 確率質量の自然対数。台の外では <see cref="double.NegativeInfinity"/>。</returns>
    double LogPmf(long k);
}
