using Tuna.LinearAlgebra;
using Tuna.SignalProcessing;
using Tuna.Statistics.Distributions;

namespace Tuna.Probabilistic.Inference;

/// <summary>
/// Computes MCMC convergence diagnostics, effective sample sizes, and Monte Carlo standard errors.
/// <para>MCMC の収束診断、有効標本数、およびモンテカルロ標準誤差を計算します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Each method accepts a single parameter's draws as a matrix with rows representing iterations and columns representing chains.
/// Split-chain diagnostics divide each chain into first and second halves, dropping the middle draw when the iteration count is odd.
/// As a rule of thumb, <c>R̂ &lt; 1.01</c> and both bulk and tail effective sample sizes of at least 100 per chain indicate reliable mixing.
/// </para>
/// <para>
/// 各メソッドは 1 つのパラメータの標本を「行 = 反復、列 = チェーン」の行列として受け取ります。
/// 分割チェーン診断では各チェーンを前半・後半に 2 分割し、反復数が奇数の場合は中央の標本を除外します。
/// 一般的な目安として、<c>R̂ &lt; 1.01</c>、かつバルクおよび裾の有効標本数がチェーン当たり 100 以上であることが望ましいとされます。
/// </para>
/// </remarks>
/// <seealso href="https://doi.org/10.1214/20-BA1221">Vehtari et al. (2021)</seealso>
public static class ConvergenceDiagnostics
{
    private const int MinimumSplitLength = 4;
    private static readonly double LowerProbability = XMath.Special.NormalCdf(-1);
    private static readonly double UpperProbability = XMath.Special.NormalCdf(1);

    /// <summary>
    /// Computes the rank-normalized split-<c>R̂</c> convergence diagnostic.
    /// <para>順位正規化した分割チェーンの収束診断 <c>R̂</c> を計算します。</para>
    /// </summary>
    /// <param name="draws">The draws matrix with rows as iterations and columns as chains. / 行を反復、列をチェーンとする標本行列。</param>
    /// <returns>The maximum of rank-normalized split-<c>R̂</c> and folded rank-normalized split-<c>R̂</c> (on absolute deviations from the median), or <see cref="double.NaN"/> if split chains have length less than 4, contain non-finite values, or are constant. / 順位正規化分割 <c>R̂</c> と中央値からの絶対偏差に対する順位正規化分割 <c>R̂</c> の大きい方。分割チェーン長が 4 未満、非有限値を含む、または全標本が定数の場合は <see cref="double.NaN"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="draws"/> is empty. / <paramref name="draws"/> が空の場合。</exception>
    public static double RHat(ReadOnlyMatrixSpan<double> draws)
    {
        Require(draws);
        if (!Usable(draws, out int half))
        {
            return double.NaN;
        }

        double[] split = Split(draws, half);
        double bulk = SplitRHatCore(RankNormalize(split), half);
        double median = Quantile(Pooled(draws), 0.5);
        double[] folded = Split(draws, half);
        for (int i = 0; i < folded.Length; i++)
        {
            folded[i] = Math.Abs(folded[i] - median);
        }

        double tail = SplitRHatCore(RankNormalize(folded), half);
        return double.IsNaN(bulk) || double.IsNaN(tail) ? double.NaN : Math.Max(bulk, tail);
    }

    /// <summary>
    /// Computes the classical split-<c>R̂</c> convergence diagnostic without rank normalization.
    /// <para>順位正規化を行わない分割チェーンの収束診断 <c>R̂</c> を計算します。</para>
    /// </summary>
    /// <param name="draws">The draws matrix with rows as iterations and columns as chains. / 行を反復、列をチェーンとする標本行列。</param>
    /// <returns>The split-<c>R̂</c> value, or <see cref="double.NaN"/> if split chains have length less than 4, contain non-finite values, or are constant. / 分割チェーンの <c>R̂</c>。分割チェーン長が 4 未満、非有限値を含む、または全標本が定数の場合は <see cref="double.NaN"/>。</returns>
    /// <remarks>
    /// <para>Intended for distributions with finite mean and variance.</para>
    /// <para>有限の平均と分散を持つ分布を対象とします。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="draws"/> is empty. / <paramref name="draws"/> が空の場合。</exception>
    public static double SplitRHat(ReadOnlyMatrixSpan<double> draws)
    {
        Require(draws);
        return Usable(draws, out int half) ? SplitRHatCore(Split(draws, half), half) : double.NaN;
    }

    /// <summary>
    /// Computes the effective sample size for estimating the posterior mean.
    /// <para>事後平均の推定に対する有効標本数を計算します。</para>
    /// </summary>
    /// <param name="draws">The draws matrix with rows as iterations and columns as chains. / 行を反復、列をチェーンとする標本行列。</param>
    /// <returns>The split-chain effective sample size, or <see cref="double.NaN"/> if split chains have length less than 4, contain non-finite values, or are constant. / 分割チェーンから求めた有効標本数。分割チェーン長が 4 未満、非有限値を含む、または全標本が定数の場合は <see cref="double.NaN"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="draws"/> is empty. / <paramref name="draws"/> が空の場合。</exception>
    public static double EffectiveSampleSize(ReadOnlyMatrixSpan<double> draws)
    {
        Require(draws);
        return Usable(draws, out int half) ? EffectiveSampleSizeCore(Split(draws, half), half) : double.NaN;
    }

    /// <summary>
    /// Computes the rank-normalized bulk effective sample size for estimating location summaries.
    /// <para>分布中心部の推定に対する順位正規化バルク有効標本数を計算します。</para>
    /// </summary>
    /// <param name="draws">The draws matrix with rows as iterations and columns as chains. / 行を反復、列をチェーンとする標本行列。</param>
    /// <returns>The bulk effective sample size, or <see cref="double.NaN"/> if split chains have length less than 4, contain non-finite values, or are constant. / 順位正規化した分割チェーンの有効標本数。分割チェーン長が 4 未満、非有限値を含む、または全標本が定数の場合は <see cref="double.NaN"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="draws"/> is empty. / <paramref name="draws"/> が空の場合。</exception>
    public static double BulkEffectiveSampleSize(ReadOnlyMatrixSpan<double> draws)
    {
        Require(draws);
        return Usable(draws, out int half) ? EffectiveSampleSizeCore(RankNormalize(Split(draws, half)), half) : double.NaN;
    }

    /// <summary>
    /// Computes the tail effective sample size for estimating extreme quantiles.
    /// <para>分布の裾の推定に対する裾有効標本数を計算します。</para>
    /// </summary>
    /// <param name="draws">The draws matrix with rows as iterations and columns as chains. / 行を反復、列をチェーンとする標本行列。</param>
    /// <returns>The minimum of the quantile effective sample sizes at 5% and 95%, or <see cref="double.NaN"/> if split chains have length less than 4, contain non-finite values, or are constant. / 5% 分位点と 95% 分位点に対する有効標本数の小さい方。分割チェーン長が 4 未満、非有限値を含む、または全標本が定数の場合は <see cref="double.NaN"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="draws"/> is empty. / <paramref name="draws"/> が空の場合。</exception>
    public static double TailEffectiveSampleSize(ReadOnlyMatrixSpan<double> draws)
    {
        Require(draws);
        if (!Usable(draws, out int half))
        {
            return double.NaN;
        }

        double[] pooled = Pooled(draws);
        double lower = IndicatorEffectiveSampleSize(draws, half, Quantile(pooled, 0.05));
        double upper = IndicatorEffectiveSampleSize(draws, half, Quantile(pooled, 0.95));
        return double.IsNaN(lower) || double.IsNaN(upper) ? double.NaN : Math.Min(lower, upper);
    }

    /// <summary>
    /// Computes the effective sample size for estimating the quantile at the specified probability.
    /// <para>指定した確率の分位点推定に対する有効標本数を計算します。</para>
    /// </summary>
    /// <param name="draws">The draws matrix with rows as iterations and columns as chains. / 行を反復、列をチェーンとする標本行列。</param>
    /// <param name="probability">The quantile probability <c>p</c> in <c>(0, 1)</c>. / 分位点の確率 <c>p</c>（<c>0 &lt; p &lt; 1</c>）。</param>
    /// <returns>The effective sample size of the indicator <c>I(θ ≤ qₚ)</c> for the pooled quantile <c>qₚ</c>, or <see cref="double.NaN"/> if split chains have length less than 4, contain non-finite values, or are constant. / 全標本の分位点 <c>qₚ</c> に対する指示関数 <c>I(θ ≤ qₚ)</c> の有効標本数。分割チェーン長が 4 未満、非有限値を含む、または全標本が定数の場合は <see cref="double.NaN"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="draws"/> is empty. / <paramref name="draws"/> が空の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="probability"/> is not in <c>(0, 1)</c>. / <paramref name="probability"/> が <c>(0, 1)</c> の範囲外の場合。</exception>
    public static double QuantileEffectiveSampleSize(ReadOnlyMatrixSpan<double> draws, double probability)
    {
        Require(draws);
        RequireProbability(probability);
        return Usable(draws, out int half) ? IndicatorEffectiveSampleSize(draws, half, Quantile(Pooled(draws), probability)) : double.NaN;
    }

    /// <summary>
    /// Computes the Monte Carlo standard error of the sample mean, <c>s / √S_eff</c>.
    /// <para>標本平均のモンテカルロ標準誤差 <c>s / √S_eff</c> を計算します。</para>
    /// </summary>
    /// <param name="draws">The draws matrix with rows as iterations and columns as chains. / 行を反復、列をチェーンとする標本行列。</param>
    /// <returns>The pooled sample standard deviation divided by the square root of <see cref="EffectiveSampleSize(ReadOnlyMatrixSpan{double})"/>, or <see cref="double.NaN"/> if unavailable. / 全標本の標準偏差を <see cref="EffectiveSampleSize(ReadOnlyMatrixSpan{double})"/> の平方根で割った値。計算不能な場合は <see cref="double.NaN"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="draws"/> is empty. / <paramref name="draws"/> が空の場合。</exception>
    public static double MeanStandardError(ReadOnlyMatrixSpan<double> draws)
    {
        Require(draws);
        if (!Usable(draws, out int half))
        {
            return double.NaN;
        }

        double ess = EffectiveSampleSizeCore(Split(draws, half), half);
        return StandardDeviation(Pooled(draws)) / Math.Sqrt(ess);
    }

    // I(θ ≤ q_p) の有効標本数 S_eff から Beta(S_eff·p + 1, S_eff·(1 − p) + 1) の Φ(±1) 分位点に対応する順序統計量差の半分を求める（Vehtari et al. 2021, §4.3）
    /// <summary>
    /// Computes the Monte Carlo standard error of the quantile at the specified probability.
    /// <para>指定した確率の分位点に対するモンテカルロ標準誤差を計算します。</para>
    /// </summary>
    /// <param name="draws">The draws matrix with rows as iterations and columns as chains. / 行を反復、列をチェーンとする標本行列。</param>
    /// <param name="probability">The quantile probability <c>p</c> in <c>(0, 1)</c>. / 分位点の確率 <c>p</c>（<c>0 &lt; p &lt; 1</c>）。</param>
    /// <returns>The estimated Monte Carlo standard error of the quantile, or <see cref="double.NaN"/> if split chains have length less than 4, contain non-finite values, or are constant. / 分位点のモンテカルロ標準誤差の推定値。分割チェーン長が 4 未満、非有限値を含む、または全標本が定数の場合は <see cref="double.NaN"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="draws"/> is empty. / <paramref name="draws"/> が空の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="probability"/> is not in <c>(0, 1)</c>. / <paramref name="probability"/> が <c>(0, 1)</c> の範囲外の場合。</exception>
    public static double QuantileStandardError(ReadOnlyMatrixSpan<double> draws, double probability)
    {
        Require(draws);
        RequireProbability(probability);
        if (!Usable(draws, out int half))
        {
            return double.NaN;
        }

        double[] sorted = Pooled(draws);
        Array.Sort(sorted);
        double ess = IndicatorEffectiveSampleSize(draws, half, QuantileOfSorted(sorted, probability));
        if (double.IsNaN(ess))
        {
            return double.NaN;
        }

        var beta = new BetaDistribution(ess * probability + 1, ess * (1 - probability) + 1);
        int size = sorted.Length;
        double lowerPosition = Math.Max(beta.InverseCdf(LowerProbability) * size - 1, 0);
        double upperPosition = Math.Min(beta.InverseCdf(UpperProbability) * size - 1, size - 1);
        return (sorted[(int)Math.Ceiling(upperPosition)] - sorted[(int)Math.Floor(lowerPosition)]) / 2;
    }

    /// <summary>
    /// Computes the sample quantile pooled across all chains using Hyndman–Fan definition 7 linear interpolation.
    /// <para>全チェーンを統合した標本の分位点を Hyndman–Fan 定義 7 の線形補間で計算します。</para>
    /// </summary>
    /// <param name="draws">The draws matrix with rows as iterations and columns as chains. / 行を反復、列をチェーンとする標本行列。</param>
    /// <param name="probability">The quantile probability <c>p</c> in <c>[0, 1]</c>. / 分位点の確率 <c>p</c>（<c>0 ≤ p ≤ 1</c>）。</param>
    /// <returns>The interpolated quantile of the pooled draws. / 全標本から線形補間された分位点。</returns>
    /// <exception cref="ArgumentException"><paramref name="draws"/> is empty. / <paramref name="draws"/> が空の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="probability"/> is not in <c>[0, 1]</c>. / <paramref name="probability"/> が <c>[0, 1]</c> の範囲外の場合。</exception>
    public static double Quantile(ReadOnlyMatrixSpan<double> draws, double probability)
    {
        Require(draws);
        if (!(probability >= 0 && probability <= 1))
        {
            throw new ArgumentOutOfRangeException(nameof(probability), "The probability must lie in [0, 1].");
        }

        return Quantile(Pooled(draws), probability);
    }

    // E-BFMI: Σ(Eₜ − Eₜ₋₁)² / Σ(Eₜ − Ē)²（要素数 2 未満なら NaN）
    internal static double EnergyFractionOfMissingInformation(ReadOnlyVectorSpan<double> energies)
    {
        int n = energies.Length;
        if (n < 2)
        {
            return double.NaN;
        }

        double mean = 0;
        for (int t = 0; t < n; t++)
        {
            mean += energies[t];
        }

        mean /= n;
        double numerator = 0, denominator = 0;
        for (int t = 0; t < n; t++)
        {
            double centered = energies[t] - mean;
            denominator += centered * centered;
            if (t > 0)
            {
                double step = energies[t] - energies[t - 1];
                numerator += step * step;
            }
        }

        return numerator / denominator;
    }

    internal static double StandardDeviation(ReadOnlySpan<double> values)
    {
        double mean = 0;
        foreach (double v in values)
        {
            mean += v;
        }

        mean /= values.Length;
        double sum = 0;
        foreach (double v in values)
        {
            double c = v - mean;
            sum += c * c;
        }

        return Math.Sqrt(sum / (values.Length - 1));
    }

    internal static double Quantile(double[] values, double probability)
    {
        double[] sorted = (double[])values.Clone();
        Array.Sort(sorted);
        return QuantileOfSorted(sorted, probability);
    }

    private static double QuantileOfSorted(double[] sorted, double probability)
    {
        double h = (sorted.Length - 1) * probability;
        int lower = (int)Math.Floor(h);
        if (lower >= sorted.Length - 1)
        {
            return sorted[^1];
        }

        return sorted[lower] + (h - lower) * (sorted[lower + 1] - sorted[lower]);
    }

    private static void Require(ReadOnlyMatrixSpan<double> draws)
    {
        if (draws.IsEmpty)
        {
            throw new ArgumentException("The draws must not be empty.", nameof(draws));
        }
    }

    private static void RequireProbability(double probability)
    {
        if (!(probability > 0 && probability < 1))
        {
            throw new ArgumentOutOfRangeException(nameof(probability), "The probability must lie in (0, 1).");
        }
    }

    private static bool Usable(ReadOnlyMatrixSpan<double> draws, out int half)
    {
        half = draws.Rows / 2;
        if (half < MinimumSplitLength)
        {
            return false;
        }

        double first = draws[0, 0];
        bool constant = true;
        for (int c = 0; c < draws.Columns; c++)
        {
            for (int t = 0; t < draws.Rows; t++)
            {
                double v = draws[t, c];
                if (!double.IsFinite(v))
                {
                    return false;
                }

                constant &= v == first;
            }
        }

        return !constant;
    }

    private static double[] Pooled(ReadOnlyMatrixSpan<double> draws)
    {
        var values = new double[draws.Rows * draws.Columns];
        int k = 0;
        for (int c = 0; c < draws.Columns; c++)
        {
            for (int t = 0; t < draws.Rows; t++)
            {
                values[k++] = draws[t, c];
            }
        }

        return values;
    }

    // 各チェーンを前半・後半に分割した 2M 本の列優先配列（長さ half、奇数行時は中央を除外）
    private static double[] Split(ReadOnlyMatrixSpan<double> draws, int half)
    {
        int n = draws.Rows;
        var split = new double[half * 2 * draws.Columns];
        for (int c = 0; c < draws.Columns; c++)
        {
            int first = 2 * c * half, second = first + half;
            for (int t = 0; t < half; t++)
            {
                split[first + t] = draws[t, c];
                split[second + t] = draws[n - half + t, c];
            }
        }

        return split;
    }

    private static double[] RankNormalize(double[] values)
    {
        int s = values.Length;
        int[] order = new int[s];
        for (int i = 0; i < s; i++)
        {
            order[i] = i;
        }

        double[] keys = (double[])values.Clone();
        Array.Sort(keys, order);
        var z = new double[s];
        int start = 0;
        while (start < s)
        {
            int end = start + 1;
            while (end < s && keys[end] == keys[start])
            {
                end++;
            }

            // 同順位には平均順位（1 始まりの start + 1, …, end の平均）を割り当てる
            double rank = (start + 1 + end) / 2.0;
            double normal = XMath.Special.InverseNormalCdf((rank - 0.375) / (s + 0.25));
            for (int i = start; i < end; i++)
            {
                z[order[i]] = normal;
            }

            start = end;
        }

        return z;
    }

    private static double SplitRHatCore(double[] split, int n)
    {
        int m = split.Length / n;
        Moments(split, n, m, out double within, out double betweenOverN, out _);
        double varPlus = (n - 1.0) / n * within + betweenOverN;
        return Math.Sqrt(varPlus / within);
    }

    // チェーン内分散 W、チェーン間分散 B/N、および各チェーンの平均を計算する
    private static void Moments(double[] split, int n, int m, out double within, out double betweenOverN, out double[] means)
    {
        means = new double[m];
        within = 0;
        for (int c = 0; c < m; c++)
        {
            ReadOnlySpan<double> chain = split.AsSpan(c * n, n);
            double mean = 0;
            foreach (double v in chain)
            {
                mean += v;
            }

            mean /= n;
            double sum = 0;
            foreach (double v in chain)
            {
                double centered = v - mean;
                sum += centered * centered;
            }

            means[c] = mean;
            within += sum / (n - 1);
        }

        within /= m;
        double grand = 0;
        foreach (double v in means)
        {
            grand += v;
        }

        grand /= m;
        double between = 0;
        foreach (double v in means)
        {
            between += (v - grand) * (v - grand);
        }

        betweenOverN = m > 1 ? between / (m - 1) : 0;
    }

    private static double IndicatorEffectiveSampleSize(ReadOnlyMatrixSpan<double> draws, int half, double threshold)
    {
        double[] split = Split(draws, half);
        bool anyBelow = false, anyAbove = false;
        for (int i = 0; i < split.Length; i++)
        {
            split[i] = split[i] <= threshold ? 1 : 0;
            anyBelow |= split[i] == 1;
            anyAbove |= split[i] == 0;
        }

        return anyBelow && anyAbove ? EffectiveSampleSizeCore(split, half) : double.NaN;
    }

    private static double EffectiveSampleSizeCore(double[] split, int n)
    {
        int m = split.Length / n;
        Moments(split, n, m, out _, out double betweenOverN, out double[] means);
        var centered = new Matrix<double>(n, m);
        for (int c = 0; c < m; c++)
        {
            for (int t = 0; t < n; t++)
            {
                centered[t, c] = split[c * n + t] - means[c];
            }
        }

        // チェーンごとの自己共分散 γ̂ₘ(t)（長さ n で割る）の、チェーンについての平均
        Matrix<double> sums = XMath.ColumnAutoCorrelations<double>(centered, n - 1);
        var autocovariance = new double[n];
        for (int t = 0; t < n; t++)
        {
            double sum = 0;
            for (int c = 0; c < m; c++)
            {
                sum += sums[t, c];
            }

            autocovariance[t] = sum / ((double)m * n);
        }

        double meanVariance = autocovariance[0] * n / (n - 1.0);
        double varPlus = meanVariance * (n - 1.0) / n + betweenOverN;
        if (!(varPlus > 0))
        {
            return double.NaN;
        }

        double Rho(int t) => t == 0 ? 1 : 1 - (meanVariance - autocovariance[t]) / varPlus;

        // Geyer の初期正列: 組 P_k = ρ̂(2k) + ρ̂(2k+1) が正の間だけ累積し、初期単調列として非増加に補正する。
        // 終了時の組（非正またはラグ n − 2 到達）は総和に含めず、その偶数項のみを加算する（非正時は正の場合のみ）
        double first = Rho(0) + Rho(1);
        double total = first, previous = first, extra = 0;
        if (first > 0 && n - 2 >= 3)
        {
            for (int k = 1; ; k++)
            {
                double even = Rho(2 * k);
                double pair = even + Rho(2 * k + 1);
                if (!(pair > 0) || 2 * k + 3 > n - 2)
                {
                    extra = pair >= 0 ? even : Math.Max(even, 0);
                    break;
                }

                previous = Math.Min(previous, pair);
                total += previous;
            }
        }

        long size = (long)n * m;
        double tau = Math.Max(-1 + 2 * total + extra, 1 / Math.Log10(size));
        return size / tau;
    }
}
