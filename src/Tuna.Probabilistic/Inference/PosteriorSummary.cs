using System.Collections;
using System.Globalization;
using System.Text;
using Tuna.LinearAlgebra;

namespace Tuna.Probabilistic.Inference;

/// <summary>
/// Represents posterior summary statistics and convergence diagnostics for a single parameter (one row of <see cref="PosteriorSummary"/>).
/// <para>単一パラメータの事後分布要約統計量および収束診断を表します（<see cref="PosteriorSummary"/> の 1 行）。</para>
/// </summary>
/// <param name="Name">The parameter name. / パラメータ名。</param>
/// <param name="Mean">The sample mean pooled across all chains. / 全チェーンを統合した標本平均。</param>
/// <param name="StandardDeviation">The unbiased sample standard deviation pooled across all chains (divisor <c>N − 1</c>). / 全チェーンを統合した不偏標本標準偏差（分母 <c>N − 1</c>）。</param>
/// <param name="MeanStandardError">The Monte Carlo standard error of the mean (<see cref="ConvergenceDiagnostics.MeanStandardError"/>). / 平均のモンテカルロ標準誤差（<see cref="ConvergenceDiagnostics.MeanStandardError"/>）。</param>
/// <param name="Quantile5">The 5th percentile (<see cref="ConvergenceDiagnostics.Quantile(ReadOnlyMatrixSpan{double}, double)"/>). / 5% 分位点（<see cref="ConvergenceDiagnostics.Quantile(ReadOnlyMatrixSpan{double}, double)"/>）。</param>
/// <param name="Median">The median (50th percentile). / 中央値（50% 分位点）。</param>
/// <param name="Quantile95">The 95th percentile. / 95% 分位点。</param>
/// <param name="BulkEffectiveSampleSize">The bulk effective sample size (<see cref="ConvergenceDiagnostics.BulkEffectiveSampleSize"/>). / バルクの有効標本数（<see cref="ConvergenceDiagnostics.BulkEffectiveSampleSize"/>）。</param>
/// <param name="TailEffectiveSampleSize">The tail effective sample size (<see cref="ConvergenceDiagnostics.TailEffectiveSampleSize"/>). / 裾の有効標本数（<see cref="ConvergenceDiagnostics.TailEffectiveSampleSize"/>）。</param>
/// <param name="RHat">The rank-normalized split-<c>R̂</c> (<see cref="ConvergenceDiagnostics.RHat"/>). / 順位正規化した分割チェーンの <c>R̂</c>（<see cref="ConvergenceDiagnostics.RHat"/>）。</param>
public sealed record ParameterSummary(
    string Name, double Mean, double StandardDeviation, double MeanStandardError,
    double Quantile5, double Median, double Quantile95,
    double BulkEffectiveSampleSize, double TailEffectiveSampleSize, double RHat);

/// <summary>
/// Represents a table of per-parameter posterior summary statistics and convergence diagnostics.
/// <para>パラメータごとの事後分布要約統計量と収束診断の表を表します。</para>
/// </summary>
/// <remarks>
/// <para><see cref="ToString"/> formats the summary as an aligned text table. See <see cref="ConvergenceDiagnostics"/> for diagnostic definitions.</para>
/// <para><see cref="ToString"/> は列を揃えたテキスト表を返します。各診断指標の定義は <see cref="ConvergenceDiagnostics"/> を参照してください。</para>
/// </remarks>
public sealed class PosteriorSummary : IReadOnlyList<ParameterSummary>
{
    private readonly ParameterSummary[] rows;

    private PosteriorSummary(ParameterSummary[] rows) => this.rows = rows;

    /// <summary>
    /// Gets the summary row for the parameter at the specified index.
    /// <para>指定したインデックスのパラメータの要約行を取得します。</para>
    /// </summary>
    /// <param name="index">The zero-based parameter index. / 0 から始まるパラメータのインデックス。</param>
    /// <value>The <see cref="ParameterSummary"/> at <paramref name="index"/>. / <paramref name="index"/> 番目の <see cref="ParameterSummary"/>。</value>
    public ParameterSummary this[int index] => rows[index];

    /// <summary>
    /// Gets the number of parameters.
    /// <para>パラメータ数を取得します。</para>
    /// </summary>
    public int Count => rows.Length;

    /// <summary>
    /// Gets the maximum <c>R̂</c> across all parameters, ignoring <see cref="double.NaN"/>.
    /// <para>全パラメータにわたる <c>R̂</c> の最大値を取得します（<see cref="double.NaN"/> は除外）。</para>
    /// </summary>
    /// <value>The maximum finite <c>R̂</c>, or <see cref="double.NaN"/> if all parameters have <see cref="double.NaN"/>. / <see cref="double.NaN"/> を除いた最大値。すべて <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</value>
    public double MaxRHat => Extreme(r => r.RHat, Math.Max);

    /// <summary>
    /// Gets the minimum of bulk and tail effective sample sizes across all parameters, ignoring <see cref="double.NaN"/>.
    /// <para>全パラメータにわたるバルクおよび裾の有効標本数の最小値を取得します（<see cref="double.NaN"/> は除外）。</para>
    /// </summary>
    /// <value>The minimum effective sample size, or <see cref="double.NaN"/> if all values are <see cref="double.NaN"/>. / <see cref="double.NaN"/> を除いた最小値。すべて <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</value>
    public double MinEffectiveSampleSize => Extreme(r => Math.Min(r.BulkEffectiveSampleSize, r.TailEffectiveSampleSize), Math.Min);

    /// <summary>
    /// Computes per-parameter posterior summary statistics and convergence diagnostics from the specified draws tensor.
    /// <para>指定した標本テンソルからパラメータごとの事後分布要約と収束診断を計算します。</para>
    /// </summary>
    /// <param name="draws">The draws tensor with shape <c>[dimension, draw, chain]</c>. / 標本を格納したテンソル（形状 <c>[パラメータ数, 反復, チェーン]</c>）。</param>
    /// <param name="names">Optional parameter names; defaults to <c>theta[0]</c>, <c>theta[1]</c>, etc. / 省略可能なパラメータ名一覧（省略時は <c>theta[0]</c>, <c>theta[1]</c>, ...）。</param>
    /// <returns>A <see cref="PosteriorSummary"/> containing one row per parameter. / 各パラメータの要約と収束診断を格納した <see cref="PosteriorSummary"/>。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="draws"/> is <see langword="null"/>. / <paramref name="draws"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="draws"/> is empty, or <paramref name="names"/> has a count different from the parameter dimension. / 標本が空の場合、または <paramref name="names"/> の要素数がパラメータ数と一致しない場合。</exception>
    public static PosteriorSummary Compute(Tensor<double> draws, IReadOnlyList<string>? names = null)
    {
        ArgumentNullException.ThrowIfNull(draws);
        int d = (int)draws.Lengths[0], n = (int)draws.Lengths[1], m = (int)draws.Lengths[2];
        if (d == 0 || n == 0 || m == 0)
        {
            throw new ArgumentException("The draws must not be empty.", nameof(draws));
        }

        if (names is not null && names.Count != d)
        {
            throw new ArgumentException($"The number of names must be {d}.", nameof(names));
        }

        var rows = new ParameterSummary[d];
        for (int i = 0; i < d; i++)
        {
            ReadOnlyMatrixSpan<double> x = SamplingResult.ParameterView(draws, i);
            double mean = 0;
            for (int c = 0; c < m; c++)
            {
                for (int t = 0; t < n; t++)
                {
                    mean += x[t, c];
                }
            }

            mean /= (double)n * m;
            double sd = n * m > 1 ? StandardDeviation(x, mean) : double.NaN;
            rows[i] = new ParameterSummary(
                names?[i] ?? $"theta[{i}]", mean, sd, ConvergenceDiagnostics.MeanStandardError(x),
                ConvergenceDiagnostics.Quantile(x, 0.05), ConvergenceDiagnostics.Quantile(x, 0.5), ConvergenceDiagnostics.Quantile(x, 0.95),
                ConvergenceDiagnostics.BulkEffectiveSampleSize(x), ConvergenceDiagnostics.TailEffectiveSampleSize(x), ConvergenceDiagnostics.RHat(x));
        }

        return new PosteriorSummary(rows);
    }

    /// <inheritdoc/>
    public IEnumerator<ParameterSummary> GetEnumerator() => ((IEnumerable<ParameterSummary>)rows).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => rows.GetEnumerator();

    /// <summary>
    /// Formats the posterior summary as an aligned text table.
    /// <para>事後分布の要約を列を揃えたテキスト表として書式設定します。</para>
    /// </summary>
    /// <returns>An aligned table formatted to 4 significant digits with effective sample sizes rounded to integers. / 有効数字 4 桁（有効標本数は整数に丸め）で列を揃えた表。</returns>
    public override string ToString()
    {
        string[] header = ["", "mean", "sd", "mcse", "5%", "50%", "95%", "ess_bulk", "ess_tail", "r_hat"];
        var table = new List<string[]> { header };
        foreach (ParameterSummary r in rows)
        {
            table.Add([
                r.Name, Format(r.Mean), Format(r.StandardDeviation), Format(r.MeanStandardError),
                Format(r.Quantile5), Format(r.Median), Format(r.Quantile95),
                FormatCount(r.BulkEffectiveSampleSize), FormatCount(r.TailEffectiveSampleSize), r.RHat.ToString("F3", CultureInfo.InvariantCulture)]);
        }

        var widths = new int[header.Length];
        foreach (string[] line in table)
        {
            for (int j = 0; j < line.Length; j++)
            {
                widths[j] = Math.Max(widths[j], line[j].Length);
            }
        }

        var builder = new StringBuilder();
        foreach (string[] line in table)
        {
            for (int j = 0; j < line.Length; j++)
            {
                builder.Append(j == 0 ? line[j].PadRight(widths[j]) : "  " + line[j].PadLeft(widths[j]));
            }

            builder.AppendLine();
        }

        return builder.ToString();
    }

    private static string Format(double value) => value.ToString("G4", CultureInfo.InvariantCulture);

    private static string FormatCount(double value) =>
        double.IsFinite(value) ? Math.Round(value).ToString("F0", CultureInfo.InvariantCulture) : value.ToString(CultureInfo.InvariantCulture);

    private static double StandardDeviation(ReadOnlyMatrixSpan<double> x, double mean)
    {
        double sum = 0;
        for (int c = 0; c < x.Columns; c++)
        {
            for (int t = 0; t < x.Rows; t++)
            {
                double centered = x[t, c] - mean;
                sum += centered * centered;
            }
        }

        return Math.Sqrt(sum / ((double)x.Rows * x.Columns - 1));
    }

    private double Extreme(Func<ParameterSummary, double> select, Func<double, double, double> combine)
    {
        double result = double.NaN;
        foreach (ParameterSummary r in rows)
        {
            double v = select(r);
            if (!double.IsNaN(v))
            {
                result = double.IsNaN(result) ? v : combine(result, v);
            }
        }

        return result;
    }
}
