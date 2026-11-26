using System.Reflection;
using Tuna.LinearAlgebra;
using Tuna.Probabilistic.Inference;

namespace Tuna.Tests.Probabilistic;

/// <summary>レプリカ交換法: 離れた峰の間の行き来、梯子の適応、尤度の焼きなまし、再現性</summary>
[Trait("Probabilistic", "Mcmc")]
public class ParallelTemperingTests
{
    private delegate void LadderAdaptation(Span<double> betas, ReadOnlySpan<double> rejections);

    private static readonly LadderAdaptation AdaptLadderPoints = typeof(ParallelTempering)
        .GetMethod("AdaptLadderPoints", BindingFlags.Static | BindingFlags.NonPublic)!.CreateDelegate<LadderAdaptation>();

    private static readonly Func<int, double, double[]> InitialLadder = typeof(ParallelTempering)
        .GetMethod("InitialLadder", BindingFlags.Static | BindingFlags.NonPublic)!.CreateDelegate<Func<int, double, double[]>>();

    [Fact]
    public void MovesBetweenSeparatedModes()
    {
        // ½N(−5·1, I) + ½N(5·1, I)。峰の間の対数密度の落ち込みは 25 で、ランダムウォークは片方の峰から出られない
        var target = new TargetDensities.TwoModes(2, 5);
        ParallelTemperingResult result = new ParallelTempering(new ParallelTemperingOptions { Seed = 71, Draws = 4000 }).Sample(target, [5.0, 5.0]);
        McmcAssert.Moments(result, [0, 0], [26, 26]);
        McmcAssert.Expectation(result, x => x[0] > 0 ? 1 : 0, 0.5, "P(x0 > 0)");
        Assert.All(result.RoundTrips, r => Assert.True(r >= 50, $"{r} round trips"));

        RandomWalkMetropolisResult stuck = new RandomWalkMetropolis(new RandomWalkMetropolisOptions { Seed = 71, Draws = 4000 }).Sample(target, [5.0, 5.0]);
        Assert.All(McmcAssert.Values(stuck, x => x[0]).AsSpan().ToArray(), v => Assert.True(v > 0));
    }

    [Fact]
    public void LadderIsAdaptedToEqualizeSwaps()
    {
        var target = new TargetDensities.TwoModes(3, 4);
        var options = new ParallelTemperingOptions { Seed = 72, Temperatures = 10, Draws = 3000, Warmup = 3000 };
        ParallelTemperingResult result = new ParallelTempering(options).Sample(target, [4.0, 4.0, 4.0]);
        Assert.Equal(10, result.InverseTemperatures.Rows);
        for (int c = 0; c < result.ChainCount; c++)
        {
            Assert.Equal(1, result.InverseTemperatures[0, c]);
            Assert.Equal(0.01, result.InverseTemperatures[9, c]);
            for (int k = 1; k < 10; k++)
            {
                Assert.True(result.InverseTemperatures[k, c] < result.InverseTemperatures[k - 1, c]);
            }

            // 隣り合う温度の交換の受理率がそろい、障壁の推定 Λ はその棄却率の和
            double[] rates = Enumerable.Range(0, 9).Select(i => result.SwapAcceptanceRates[i, c]).ToArray();
            Assert.True(rates.Max() - rates.Min() < 0.2, string.Join(", ", rates));
            Assert.Equal(rates.Sum(r => 1 - r), result.CommunicationBarriers[c], 12);
            for (int k = 0; k < 10; k++)
            {
                Assert.InRange(result.LocalAcceptanceRates[k, c], 0.05, 0.6);
            }
        }

        McmcAssert.Moments(result, [0, 0, 0], [17, 17, 17]);
    }

    [Fact]
    public void TemperingTheLikelihoodSamplesThePosterior()
    {
        // 事前分布 N(0, 10²I) と二峰の尤度。事後分布は N(±5·100/101·1, 100/101·I) の等しい重みの混合
        var likelihood = new TargetDensities.TwoModes(2, 5);
        double LogPrior(ReadOnlyVectorSpan<double> x) => -0.005 * (x[0] * x[0] + x[1] * x[1]);
        ParallelTemperingResult result = new ParallelTempering(new ParallelTemperingOptions { Seed = 73, Draws = 4000 })
            .Sample(LogPrior, x => likelihood.LogDensity(x), [5.0, 5.0]);
        double center = 500.0 / 101, spread = 100.0 / 101;
        McmcAssert.Moments(result, [0, 0], [spread + center * center, spread + center * center]);
        for (int c = 0; c < result.ChainCount; c++)
        {
            Assert.Equal(0, result.InverseTemperatures[7, c]);
        }

        // 対数密度は β = 1 の事前分布と尤度の和
        double[] point = [result.Draws[0, 10, 2], result.Draws[1, 10, 2]];
        Assert.Equal(LogPrior(point) + likelihood.LogDensity(point), result.LogDensities[10, 2], 12);
    }

    [Fact]
    public void PriorOutsideTheSupportSkipsTheLikelihood()
    {
        int likelihoodCalls = 0;
        double LogPrior(ReadOnlyVectorSpan<double> x) => x[0] >= 0 && x[0] <= 1 ? 0 : double.NegativeInfinity;
        double LogLikelihood(ReadOnlyVectorSpan<double> x)
        {
            Assert.InRange(x[0], 0, 1);
            Interlocked.Increment(ref likelihoodCalls);
            return 3 * Math.Log(x[0]);
        }

        // 事後分布 Beta(4, 1): 平均 4/5、分散 4/150
        ParallelTemperingResult result = new ParallelTempering(new ParallelTemperingOptions { Seed = 74, Temperatures = 4, Draws = 3000 })
            .Sample(LogPrior, LogLikelihood, [0.5]);
        McmcAssert.Moments(result, [0.8], [4.0 / 150]);
        Assert.True(likelihoodCalls < result.TotalEvaluations);
    }

    [Fact]
    public void ResultsAreReproducible()
    {
        var target = new TargetDensities.TwoModes(2, 3);
        var options = new ParallelTemperingOptions { Seed = 75, Temperatures = 5, Draws = 200, Warmup = 300 };
        ParallelTemperingResult a = new ParallelTempering(options).Sample(target, [3.0, 3.0]);
        ParallelTemperingResult b = new ParallelTempering(new ParallelTemperingOptions { Seed = 75, Temperatures = 5, Draws = 200, Warmup = 300, MaxDegreeOfParallelism = 4 })
            .Sample(x => target.LogDensity(x), [3.0, 3.0]);
        McmcAssert.Identical(a, b);
        Assert.True(a.InverseTemperatures.AsSpan().SequenceEqual(b.InverseTemperatures.AsSpan()));
        Assert.True(a.RoundTrips.SequenceEqual(b.RoundTrips));

        var starts = new Matrix<double>(new[,] { { 3.0, -3.0, 3.0, -3.0 }, { 3.0, -3.0, -3.0, 3.0 } });
        ParallelTemperingResult fromColumns = new ParallelTempering(options).Sample(target, starts);
        Assert.Equal(4, fromColumns.ChainCount);
        Assert.All(fromColumns.Evaluations, e => Assert.Equal(1 + 5 * (300 + 200), e));
    }

    [Fact]
    public void FixedLadderStaysGeometric()
    {
        var options = new ParallelTemperingOptions { Seed = 76, Temperatures = 5, MinimumInverseTemperature = 0.0625, AdaptLadder = false, Draws = 100, Chains = 1 };
        ParallelTemperingResult result = new ParallelTempering(options).Sample(new TargetDensities.TwoModes(1, 2), [0.0]);
        double[] expected = [1, 0.5, 0.25, 0.125, 0.0625];
        for (int k = 0; k < 5; k++)
        {
            Assert.Equal(expected[k], result.InverseTemperatures[k, 0], 14);
        }
    }

    [Fact]
    public void LadderPointsFollowTheCumulativeRejection()
    {
        Assert.Equal(new[] { 1.0, 0.75, 0.5, 0.25, 0.0 }, InitialLadder(5, 0));
        double[] geometric = InitialLadder(3, 0.01);
        Assert.Equal(0.1, geometric[1], 14);

        // 棄却率がそろっていれば梯子は変わらない
        double[] betas = [1, 0.75, 0.5, 0.25, 0];
        AdaptLadderPoints(betas, [0.3, 0.3, 0.3, 0.3]);
        Assert.Equal(new[] { 1.0, 0.75, 0.5, 0.25, 0.0 }, betas.Select(b => Math.Round(b, 12)));

        // 棄却が最後の区間 (0.25, 0) に集中していれば、中間の点はそこへ寄る
        AdaptLadderPoints(betas, [0.0, 0.0, 0.0, 0.9]);
        Assert.Equal(1, betas[0]);
        Assert.Equal(0, betas[4]);
        Assert.True(betas[1] < 0.25 && betas[1] > betas[2] && betas[2] > betas[3] && betas[3] > 0, string.Join(", ", betas));

        // 推定できなかった区間（NaN）は棄却がほとんどないものとして扱う
        double[] withNaN = [1, 0.5, 0.25, 0];
        AdaptLadderPoints(withNaN, [double.NaN, 0.5, 0.5]);
        Assert.True(withNaN[1] < 0.5 && withNaN[1] > withNaN[2] && withNaN[2] > 0);
    }

    [Fact]
    public void OptionsAreValidated()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ParallelTempering(new ParallelTemperingOptions { Temperatures = 1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ParallelTempering(new ParallelTemperingOptions { MinimumInverseTemperature = 1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ParallelTempering(new ParallelTemperingOptions { MinimumInverseTemperature = -0.1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ParallelTempering(new ParallelTemperingOptions { LocalSteps = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ParallelTempering(new ParallelTemperingOptions { TargetAcceptance = 1.5 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ParallelTempering(new ParallelTemperingOptions { Chains = 0 }));

        // 密度全体を焼きなます場合、β = 0 の分布は正規化できない
        var flat = new ParallelTempering(new ParallelTemperingOptions { MinimumInverseTemperature = 0 });
        Assert.Throws<ArgumentOutOfRangeException>(() => flat.Sample(new TargetDensities.TwoModes(1, 1), [0.0]));
        Assert.Throws<ArgumentException>(() => new ParallelTempering().Sample(new TargetDensities.UnitBox(), [2.0]));
        Assert.Throws<ArgumentException>(() => new ParallelTempering().Sample(new TargetDensities.UnitBox(), ReadOnlySpan<double>.Empty));
        Assert.Throws<ArgumentException>(() => new ParallelTempering().Sample(new TargetDensities.UnitBox(), new Matrix<double>(1, 3)));
    }
}
