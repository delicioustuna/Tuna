using Tuna.LinearAlgebra;
using Tuna.Probabilistic.Inference;

namespace Tuna.Tests.Probabilistic;

/// <summary>
/// 標本による期待値の推定を、閉じた形の値とモンテカルロ標準誤差（<see cref="ConvergenceDiagnostics.MeanStandardError"/>）で比べる。
/// シードを固定するので結果は決定的で、許容幅 5·MCSE は「その推定がたまたま外れる」確率がおよそ 6×10⁻⁷ の幅になる
/// </summary>
internal static class McmcAssert
{
    internal const double Sigmas = 5;

    /// <summary>関数 <paramref name="f"/> の値の <c>反復 × チェーン</c> の行列</summary>
    internal static Matrix<double> Values(SamplingResult result, Func<double[], double> f)
    {
        int n = result.DrawCount, m = result.ChainCount, d = result.Dimension;
        var values = new Matrix<double>(n, m);
        var point = new double[d];
        for (int c = 0; c < m; c++)
        {
            MatrixSpan<double> chain = result.Chain(c);
            for (int t = 0; t < n; t++)
            {
                for (int i = 0; i < d; i++)
                {
                    point[i] = chain[i, t];
                }

                values[t, c] = f(point);
            }
        }

        return values;
    }

    /// <summary>E[f(θ)] の推定が <paramref name="expected"/> から 5·MCSE 以内で、有効標本数が十分にあり、R̂ が 1.05 未満であること</summary>
    internal static void Expectation(SamplingResult result, Func<double[], double> f, double expected, string label, double minimumEss = 100)
    {
        Matrix<double> values = Values(result, f);
        double mean = 0;
        foreach (double v in values.AsSpan())
        {
            mean += v;
        }

        mean /= values.Rows * values.Columns;
        double mcse = ConvergenceDiagnostics.MeanStandardError(values);
        double ess = ConvergenceDiagnostics.EffectiveSampleSize(values);
        double rhat = ConvergenceDiagnostics.RHat(values);
        Assert.True(ess >= minimumEss, $"{label}: effective sample size {ess} < {minimumEss}");
        Assert.True(rhat < 1.05, $"{label}: R-hat {rhat}");
        Assert.True(Math.Abs(mean - expected) <= Sigmas * mcse,
            $"{label}: estimate {mean} (MCSE {mcse}) differs from {expected} by {Math.Abs(mean - expected) / mcse:F1} standard errors");
    }

    /// <summary>各成分の平均と分散（中心は既知の平均）</summary>
    internal static void Moments(SamplingResult result, double[] mean, double[] variance, double minimumEss = 100)
    {
        for (int i = 0; i < mean.Length; i++)
        {
            int k = i;
            Expectation(result, x => x[k], mean[k], $"E[x{k}]", minimumEss);
            Expectation(result, x => (x[k] - mean[k]) * (x[k] - mean[k]), variance[k], $"Var[x{k}]", minimumEss);
        }
    }

    /// <summary>平均と共分散行列のすべての成分</summary>
    internal static void Moments(SamplingResult result, double[] mean, Matrix<double> covariance, double minimumEss = 100)
    {
        for (int i = 0; i < mean.Length; i++)
        {
            int a = i;
            Expectation(result, x => x[a], mean[a], $"E[x{a}]", minimumEss);
            for (int j = 0; j <= i; j++)
            {
                int b = j;
                Expectation(result, x => (x[a] - mean[a]) * (x[b] - mean[b]), covariance[a, b], $"Cov[x{a}, x{b}]", minimumEss);
            }
        }
    }

    /// <summary>2つの結果の標本と対数密度がビット単位で一致すること</summary>
    internal static void Identical(SamplingResult expected, SamplingResult actual)
    {
        Assert.True(expected.Draws.AsSpan().SequenceEqual(actual.Draws.AsSpan()), "draws differ");
        Assert.True(expected.LogDensities.AsSpan().SequenceEqual(actual.LogDensities.AsSpan()), "log densities differ");
        Assert.Equal(expected.Evaluations, actual.Evaluations);
    }
}
