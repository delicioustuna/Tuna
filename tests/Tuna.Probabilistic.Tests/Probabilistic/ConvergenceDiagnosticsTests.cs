using Tuna.LinearAlgebra;
using Tuna.Probabilistic.Inference;

namespace Tuna.Tests.Probabilistic;

/// <summary>
/// 収束の診断を ArviZ の値（tools/McmcDiagnosticsReferences が生成）と照合し、定義の境界（短い・一定・有限でない標本）を確かめる
/// </summary>
[Trait("Probabilistic", "Mcmc")]
public class ConvergenceDiagnosticsTests
{
    public static TheoryData<string, string, double, double> References()
    {
        var data = new TheoryData<string, string, double, double>();
        foreach (var (name, function, probability, expected) in DiagnosticsReferences.Values)
        {
            data.Add(name, function, probability, expected);
        }

        return data;
    }

    /// <summary>参照値の生成と同じ手順で標本 <c>反復 × チェーン</c> を作る（線形合同法と四則演算だけなので Python と丸めまで一致する）</summary>
    internal static Matrix<double> Generate(string name)
    {
        var c = DiagnosticsReferences.Cases.Single(x => x.Name == name);
        var draws = new Matrix<double>(c.Draws, c.Chains);
        ulong state = c.Seed;
        double Uniform()
        {
            state = unchecked(state * 6364136223846793005UL + 1442695040888963407UL);
            return ((state >> 11) + 0.5) * Math.Pow(2, -53);
        }

        for (int m = 0; m < c.Chains; m++)
        {
            double x = 0;
            for (int t = 0; t < c.Draws; t++)
            {
                double e;
                if (c.Kind == "ratio")
                {
                    double u1 = Uniform();
                    double u2 = Uniform();
                    e = (u1 - 0.5) / u2;
                }
                else
                {
                    e = Uniform() - 0.5;
                }

                x = c.Phi * x + e;
                double value = c.Scales[m] * x + c.Shifts[m];
                if (c.Kind == "ties")
                {
                    value = Math.Floor(value * 2.0) / 2.0;
                }

                draws[t, m] = value;
            }
        }

        return draws;
    }

    [Theory]
    [MemberData(nameof(References))]
    public void MatchesArviZ(string name, string function, double probability, double expected)
    {
        Matrix<double> draws = Generate(name);
        double actual = function switch
        {
            "RHat" => ConvergenceDiagnostics.RHat(draws),
            "SplitRHat" => ConvergenceDiagnostics.SplitRHat(draws),
            "EffectiveSampleSize" => ConvergenceDiagnostics.EffectiveSampleSize(draws),
            "BulkEffectiveSampleSize" => ConvergenceDiagnostics.BulkEffectiveSampleSize(draws),
            "TailEffectiveSampleSize" => ConvergenceDiagnostics.TailEffectiveSampleSize(draws),
            "QuantileEffectiveSampleSize" => ConvergenceDiagnostics.QuantileEffectiveSampleSize(draws, probability),
            "MeanStandardError" => ConvergenceDiagnostics.MeanStandardError(draws),
            "QuantileStandardError" => ConvergenceDiagnostics.QuantileStandardError(draws, probability),
            "Quantile" => ConvergenceDiagnostics.Quantile(draws, probability),
            _ => throw new ArgumentException(function),
        };

        Assert.True(Math.Abs(actual - expected) <= DiagnosticsReferences.Tolerance * Math.Abs(expected),
            $"{name} {function}({probability}): expected {expected:R}, actual {actual:R}");
    }

    [Fact]
    public void SeparatedChainsGiveLargeRHat()
    {
        // 平均のずれたチェーン（ShiftedChain）と尺度の違うチェーン（ScaledChain）は R̂ が 1.01 を大きく超える。
        // 尺度の違いは順位正規化した R̂ ではなく、中央値からの距離（折り返し）の R̂ が捉える
        Assert.True(ConvergenceDiagnostics.RHat(Generate("ShiftedChain")) > 1.05);
        Matrix<double> scaled = Generate("ScaledChain");
        Assert.True(ConvergenceDiagnostics.RHat(scaled) > 1.05);
        Assert.True(ConvergenceDiagnostics.SplitRHat(scaled) < 1.02);
        Assert.True(ConvergenceDiagnostics.RHat(Generate("Independent")) < 1.01);
    }

    [Fact]
    public void AntitheticChainsAreBoundedByLogTen()
    {
        // 逆相関の連鎖では τ̂ が 1 を下回り、有効標本数は標本数を超えるが、τ̂ ≥ 1/log₁₀ S で抑える
        Matrix<double> draws = Generate("Antithetic");
        double ess = ConvergenceDiagnostics.EffectiveSampleSize(draws);
        Assert.True(ess > draws.Rows * draws.Columns);
        Assert.True(ess <= draws.Rows * draws.Columns * Math.Log10(draws.Rows * draws.Columns) * (1 + 1e-12));
    }

    [Fact]
    public void SingleAndOddLengthChainsAreFinite()
    {
        // ArviZ は 1 本のチェーンの R̂ を NaN にし、奇数長の折り返しの中央値に真ん中の標本を含めないが、
        // 本ライブラリは Vehtari et al. 2021・R の posterior と同じく、分割チェーンの R̂ を計算し、全標本の中央値で折り返す
        double single = ConvergenceDiagnostics.RHat(Generate("SingleChain"));
        Assert.InRange(single, 0.99, 1.05);
        double odd = ConvergenceDiagnostics.RHat(Generate("OddLength"));
        Assert.InRange(odd, 0.99, 1.02);
    }

    [Fact]
    public void ReturnsNaNForUnusableDraws()
    {
        var shortDraws = new Matrix<double>(7, 2);
        for (int t = 0; t < 7; t++)
        {
            shortDraws[t, 0] = t;
            shortDraws[t, 1] = -t;
        }

        var constant = new Matrix<double>(100, 2);
        constant.AsSpan().Fill(3);
        Matrix<double> nonFinite = Generate("Independent");
        nonFinite[10, 2] = double.NaN;
        foreach (Matrix<double> draws in new[] { shortDraws, constant, nonFinite })
        {
            Assert.True(double.IsNaN(ConvergenceDiagnostics.RHat(draws)));
            Assert.True(double.IsNaN(ConvergenceDiagnostics.SplitRHat(draws)));
            Assert.True(double.IsNaN(ConvergenceDiagnostics.EffectiveSampleSize(draws)));
            Assert.True(double.IsNaN(ConvergenceDiagnostics.BulkEffectiveSampleSize(draws)));
            Assert.True(double.IsNaN(ConvergenceDiagnostics.TailEffectiveSampleSize(draws)));
            Assert.True(double.IsNaN(ConvergenceDiagnostics.MeanStandardError(draws)));
            Assert.True(double.IsNaN(ConvergenceDiagnostics.QuantileStandardError(draws, 0.5)));
        }

        // 指示関数 I(θ ≤ q_p) がすべて同じ値なら（離散の値で q_p が最大値に一致する場合）分位点の有効標本数・MCSE は NaN（ArviZ は S・0 を返す）
        Matrix<double> ties = Generate("Ties");
        Assert.True(double.IsNaN(ConvergenceDiagnostics.QuantileEffectiveSampleSize(ties, 0.95)));
        Assert.True(double.IsNaN(ConvergenceDiagnostics.QuantileStandardError(ties, 0.95)));
        Assert.True(double.IsNaN(ConvergenceDiagnostics.TailEffectiveSampleSize(ties)));

        // 長さ 8（分割チェーンの長さ 4）なら計算する
        var minimal = new Matrix<double>(8, 2);
        for (int t = 0; t < 8; t++)
        {
            minimal[t, 0] = Math.Sin(t);
            minimal[t, 1] = Math.Cos(t);
        }

        Assert.True(double.IsFinite(ConvergenceDiagnostics.RHat(minimal)));
        Assert.True(double.IsFinite(ConvergenceDiagnostics.EffectiveSampleSize(minimal)));
    }

    [Fact]
    public void ValidatesArguments()
    {
        Matrix<double> draws = Generate("Independent");
        Assert.Throws<ArgumentException>(() => ConvergenceDiagnostics.RHat(new Matrix<double>(0, 4)));
        Assert.Throws<ArgumentException>(() => ConvergenceDiagnostics.EffectiveSampleSize(new Matrix<double>(10, 0)));
        Assert.Throws<ArgumentOutOfRangeException>(() => ConvergenceDiagnostics.QuantileEffectiveSampleSize(draws, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => ConvergenceDiagnostics.QuantileStandardError(draws, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => ConvergenceDiagnostics.Quantile(draws, 1.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => ConvergenceDiagnostics.Quantile(draws, double.NaN));
        Assert.Equal(ConvergenceDiagnostics.Quantile(draws, 0), draws.AsSpan().ToArray().Min());
        Assert.Equal(ConvergenceDiagnostics.Quantile(draws, 1), draws.AsSpan().ToArray().Max());
    }

    [Fact]
    public void SummaryUsesTheSameDiagnostics()
    {
        Matrix<double> a = Generate("Autocorrelated"), b = Generate("HeavyTails");
        int n = a.Rows, m = a.Columns;
        var draws = new Tensor<double>(2, n, m);
        for (int c = 0; c < m; c++)
        {
            for (int t = 0; t < n; t++)
            {
                draws[0, t, c] = a[t, c];
                draws[1, t, c] = b[t, c];
            }
        }

        PosteriorSummary summary = PosteriorSummary.Compute(draws, ["a", "b"]);
        Assert.Equal(2, summary.Count);
        Assert.Equal("a", summary[0].Name);
        Assert.Equal(ConvergenceDiagnostics.RHat(a), summary[0].RHat);
        Assert.Equal(ConvergenceDiagnostics.BulkEffectiveSampleSize(b), summary[1].BulkEffectiveSampleSize);
        Assert.Equal(ConvergenceDiagnostics.TailEffectiveSampleSize(b), summary[1].TailEffectiveSampleSize);
        Assert.Equal(ConvergenceDiagnostics.MeanStandardError(a), summary[0].MeanStandardError);
        Assert.Equal(ConvergenceDiagnostics.Quantile(b, 0.5), summary[1].Median);
        Assert.Equal(Math.Max(summary[0].RHat, summary[1].RHat), summary.MaxRHat);
        Assert.Contains("ess_bulk", summary.ToString());
        Assert.Throws<ArgumentException>(() => PosteriorSummary.Compute(draws, ["only one"]));
    }
}
