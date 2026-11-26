using Tuna.LinearAlgebra;
using Tuna.Optimization.Global;
using Tuna.Probabilistic.Inference;

namespace Tuna.Tests.Probabilistic;

/// <summary>NUTS と静的な HMC: 閉じた形の平均・共分散との照合、調整、診断量、再現性、引数の検証</summary>
[Trait("Probabilistic", "Mcmc")]
public class HamiltonianSamplerTests
{
    private static readonly double[] IidMean = [0, 1, 2, 3, 4];
    private static readonly double[] IidSd = [1, 0.1, 10, 3, 0.5];

    private static double[] Squares(double[] sd) => sd.Select(s => s * s).ToArray();

    [Theory]
    [InlineData(HamiltonianMetric.Diagonal)]
    [InlineData(HamiltonianMetric.Dense)]
    public void NutsRecoversIndependentNormals(HamiltonianMetric metric)
    {
        var sampler = new NoUTurnSampler(new NoUTurnSamplerOptions { Seed = 11, Metric = metric });
        HamiltonianSamplingResult result = sampler.Sample(new TargetDensities.IndependentNormal(IidMean, IidSd), new double[5]);
        McmcAssert.Moments(result, IidMean, Squares(IidSd));
        Assert.Equal(0, result.DivergenceCount);
        Assert.Equal(0, result.MaxTreeDepthHits);
        for (int c = 0; c < result.ChainCount; c++)
        {
            Assert.True(result.EnergyBayesianFractionOfMissingInformation(c) > 0.3);

            // 計量の逆は目標の分散（対角）に近づく
            for (int i = 0; i < 5; i++)
            {
                Assert.InRange(result.InverseMetrics[i, i, c] / (IidSd[i] * IidSd[i]), 0.5, 2);
            }
        }

        double acceptance = result.AcceptanceStatistics.AsSpan().ToArray().Average();
        Assert.InRange(acceptance, 0.7, 0.95);
    }

    [Fact]
    public void DenseMetricRemovesCorrelation()
    {
        var target = new TargetDensities.CorrelatedNormal(1, -2, 1, 10, 0.95);
        HamiltonianSamplingResult dense = new NoUTurnSampler(new NoUTurnSamplerOptions { Seed = 12, Metric = HamiltonianMetric.Dense })
            .Sample(target, [0.0, 0.0]);
        HamiltonianSamplingResult diagonal = new NoUTurnSampler(new NoUTurnSamplerOptions { Seed = 12 }).Sample(target, [0.0, 0.0]);
        McmcAssert.Moments(dense, [1, -2], target.Covariance);
        McmcAssert.Moments(diagonal, [1, -2], target.Covariance);

        // 全体の計量では相関を取り除くので、1回の遷移のリープフロッグの回数が少ない
        double denseSteps = dense.LeapfrogSteps.AsSpan().ToArray().Average();
        double diagonalSteps = diagonal.LeapfrogSteps.AsSpan().ToArray().Average();
        Assert.True(denseSteps < 0.6 * diagonalSteps, $"dense {denseSteps}, diagonal {diagonalSteps}");
        Assert.InRange(dense.InverseMetrics[0, 1, 0] / (0.95 * 10), 0.5, 1.5);
        Assert.Equal(0, diagonal.InverseMetrics[0, 1, 0]);
    }

    [Fact]
    public void NutsFollowsCurvedTarget()
    {
        var target = new TargetDensities.Banana(3, 0.01);
        HamiltonianSamplingResult result = new NoUTurnSampler(new NoUTurnSamplerOptions { Seed = 13, Draws = 2000 })
            .Sample(target, new double[3]);
        McmcAssert.Moments(result, [0, 0, 0], [100, target.Variance2, 1]);

        // 裾（|x₁| ≈ 30）では曲率が中心の数十倍になり、中心に合わせたステップ幅ではまれに発散する
        Assert.True(result.DivergenceCount < 0.005 * result.DrawCount * result.ChainCount, $"{result.DivergenceCount} divergences");
    }

    [Fact]
    public void FunnelProducesDivergences()
    {
        // 漏斗の首（v が小さい所）では固定のステップ幅が大きすぎ、軌道が発散する。発散は記録され、標本は有限のまま
        HamiltonianSamplingResult result = new NoUTurnSampler(new NoUTurnSamplerOptions { Seed = 14 })
            .Sample(new TargetDensities.Funnel(), [0.0, 0.0]);
        Assert.True(result.DivergenceCount > 0);
        Assert.All(result.Draws.AsSpan().ToArray(), v => Assert.True(double.IsFinite(v)));
        int marked = result.Divergences.AsSpan().ToArray().Count(v => v == 1);
        Assert.Equal(result.DivergenceCount, marked);
    }

    [Fact]
    public void TreeDepthIsBounded()
    {
        // 尺度の大きく違う目標で計量を適応させないと、深い木が要る。上限で打ち切って記録する
        var options = new NoUTurnSamplerOptions { Seed = 15, MaxTreeDepth = 3, AdaptMetric = false, Draws = 200, Chains = 2 };
        HamiltonianSamplingResult result = new NoUTurnSampler(options)
            .Sample(new TargetDensities.IndependentNormal([0, 0], [0.01, 100]), [0.0, 0.0]);
        Assert.Equal(3, result.MaxTreeDepth);
        Assert.All(result.TreeDepths.AsSpan().ToArray(), v => Assert.InRange(v, 1, 3));
        Assert.True(result.MaxTreeDepthHits > 0);
        Assert.All(result.LeapfrogSteps.AsSpan().ToArray(), v => Assert.InRange(v, 1, 7));
    }

    [Fact]
    public void StaticHmcRecoversIndependentNormals()
    {
        var options = new HamiltonianMonteCarloOptions { Seed = 16, Steps = 20, StepSizeJitter = 0.2, Draws = 2000 };
        HamiltonianSamplingResult result = new HamiltonianMonteCarlo(options)
            .Sample(new TargetDensities.IndependentNormal(IidMean, IidSd), new double[5]);
        McmcAssert.Moments(result, IidMean, Squares(IidSd));
        Assert.All(result.TreeDepths.AsSpan().ToArray(), v => Assert.Equal(0, v));
        Assert.Equal(0, result.MaxTreeDepth);
        Assert.All(result.LeapfrogSteps.AsSpan().ToArray(), v => Assert.Equal(20, v));
        double acceptance = result.AcceptanceStatistics.AsSpan().ToArray().Average();
        Assert.InRange(acceptance, 0.5, 0.85);
    }

    [Fact]
    public void GradientsFromPotentialsWork()
    {
        // 手書きの勾配の代わりに、−ln π を分解可能な関数として書いた目的関数から勾配を作る（GradientDual で値と勾配を1回で計算）
        var potential = new FactorableObjective<CorrelatedNormalPotential>(new CorrelatedNormalPotential(), 2);
        HamiltonianSamplingResult result = new NoUTurnSampler(new NoUTurnSamplerOptions { Seed = 17 })
            .Sample(LogDensities.FromPotential(potential), [0.0, 0.0]);
        McmcAssert.Moments(result, [1, -2], new TargetDensities.CorrelatedNormal(1, -2, 1, 10, 0.9).Covariance);
    }

    /// <summary>平均 (1, −2)、標準偏差 (1, 10)、相関 0.9 の正規分布のポテンシャル −ln π（定数を除く）</summary>
    private readonly struct CorrelatedNormalPotential : IFactorableFunction
    {
        public T Evaluate<T>(ReadOnlySpan<T> x) where T : IFactorableNumber<T>
        {
            T a = x[0] - 1.0;
            T b = (x[1] + 2.0) * 0.1;
            return 0.5 / (1 - 0.81) * (a * a - 1.8 * a * b + b * b);
        }
    }

    [Fact]
    public void ResultsAreReproducible()
    {
        var target = new TargetDensities.CorrelatedNormal(1, -2, 1, 10, 0.9);
        var options = new NoUTurnSamplerOptions { Seed = 18, Draws = 200, Warmup = 200 };
        HamiltonianSamplingResult sequential = new NoUTurnSampler(options).Sample(target, [0.0, 0.0]);
        HamiltonianSamplingResult parallel = new NoUTurnSampler(new NoUTurnSamplerOptions { Seed = 18, Draws = 200, Warmup = 200, MaxDegreeOfParallelism = -1 })
            .Sample(target, [0.0, 0.0]);
        McmcAssert.Identical(sequential, parallel);
        Assert.True(sequential.StepSizes.SequenceEqual(parallel.StepSizes));
        Assert.Equal(18UL, sequential.Seed);

        // デリゲートで渡しても同じ
        HamiltonianSamplingResult viaDelegate = new NoUTurnSampler(options).Sample((x, g) => target.LogDensity(x, g), [0.0, 0.0]);
        McmcAssert.Identical(sequential, viaDelegate);

        // 間引くと、間引かない標本列の Thin 個ごとの最後の標本になる
        HamiltonianSamplingResult thinned = new NoUTurnSampler(new NoUTurnSamplerOptions { Seed = 18, Draws = 100, Warmup = 200, Thin = 2 })
            .Sample(target, [0.0, 0.0]);
        for (int c = 0; c < 4; c++)
        {
            for (int t = 0; t < 100; t++)
            {
                Assert.Equal(sequential.Draws[0, 2 * t + 1, c], thinned.Draws[0, t, c]);
            }
        }

        HamiltonianSamplingResult other = new NoUTurnSampler(new NoUTurnSamplerOptions { Seed = 19, Draws = 200, Warmup = 200 }).Sample(target, [0.0, 0.0]);
        Assert.False(sequential.Draws.AsSpan().SequenceEqual(other.Draws.AsSpan()));
    }

    [Fact]
    public void FixedTuningIsUsedWithoutWarmup()
    {
        var inverseMetric = new Matrix<double>(new[,] { { 1.0, 0.9 * 10 }, { 0.9 * 10, 100.0 } });
        var options = new NoUTurnSamplerOptions
        {
            Seed = 20,
            Warmup = 0,
            Draws = 2000,
            InitialStepSize = 0.7,
            Metric = HamiltonianMetric.Dense,
            InitialInverseMetric = inverseMetric,
        };
        var target = new TargetDensities.CorrelatedNormal(1, -2, 1, 10, 0.9);
        HamiltonianSamplingResult result = new NoUTurnSampler(options).Sample(target, [1.0, -2.0]);
        Assert.All(result.StepSizes, s => Assert.Equal(0.7, s));
        Assert.Equal(9, result.InverseMetrics[1, 0, 3]);
        McmcAssert.Moments(result, [1, -2], target.Covariance);
    }

    [Fact]
    public void ChainsCanStartFromDifferentPoints()
    {
        var starts = new Matrix<double>(new[,] { { -5.0, 5.0, 0.0 }, { 20.0, -20.0, 0.0 } });
        var options = new NoUTurnSamplerOptions { Seed = 21, Chains = 3 };
        var target = new TargetDensities.CorrelatedNormal(1, -2, 1, 10, 0.9);
        HamiltonianSamplingResult result = new NoUTurnSampler(options).Sample(target, starts);
        Assert.Equal(3, result.ChainCount);
        McmcAssert.Moments(result, [1, -2], target.Covariance);
        Assert.Throws<ArgumentException>(() => new NoUTurnSampler(new NoUTurnSamplerOptions { Chains = 2 }).Sample(target, starts));
    }

    [Fact]
    public void InvalidStartsAndDensitiesThrow()
    {
        var sampler = new NoUTurnSampler(new NoUTurnSamplerOptions { Seed = 22, Warmup = 10, Draws = 10 });
        var target = new TargetDensities.IndependentNormal([0.0], [1.0]);
        Assert.Throws<ArgumentException>(() => sampler.Sample(target, ReadOnlySpan<double>.Empty));
        Assert.Throws<ArgumentException>(() => sampler.Sample(target, [double.NaN]));
        Assert.Throws<ArgumentException>(() => sampler.Sample((x, g) => double.NegativeInfinity, [0.0]));
        Assert.Throws<ArgumentNullException>(() => sampler.Sample((Func<ReadOnlyVectorSpan<double>, VectorSpan<double>, double>)null!, [0.0]));
        var wrongMetric = new NoUTurnSampler(new NoUTurnSamplerOptions { InitialInverseMetric = new Matrix<double>(3, 3) });
        Assert.Throws<ArgumentException>(() => wrongMetric.Sample(target, [0.0]));

        // 対数密度の例外は、並列に動かしても包まずにそのまま伝わる
        var throwing = new NoUTurnSampler(new NoUTurnSamplerOptions { Seed = 22, MaxDegreeOfParallelism = 4 });
        Assert.Throws<InvalidOperationException>(() => throwing.Sample((x, g) => x[0] > 0.5 ? throw new InvalidOperationException() : -0.5 * x[0] * x[0], [0.0]));
    }

    [Fact]
    public void SamplingCanBeCanceled()
    {
        using var source = new CancellationTokenSource();
        source.Cancel();
        var target = new TargetDensities.IndependentNormal([0.0], [1.0]);
        Assert.ThrowsAny<OperationCanceledException>(() => new NoUTurnSampler().Sample(target, [0.0], source.Token));
        Assert.ThrowsAny<OperationCanceledException>(() =>
            new HamiltonianMonteCarlo(new HamiltonianMonteCarloOptions { MaxDegreeOfParallelism = 2 }).Sample(target, [0.0], source.Token));
    }

    [Fact]
    public void OptionsAreValidated()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new NoUTurnSampler(new NoUTurnSamplerOptions { MaxTreeDepth = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NoUTurnSampler(new NoUTurnSamplerOptions { MaxTreeDepth = 31 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NoUTurnSampler(new NoUTurnSamplerOptions { TargetAcceptance = 1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NoUTurnSampler(new NoUTurnSamplerOptions { InitialStepSize = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NoUTurnSampler(new NoUTurnSamplerOptions { DivergenceThreshold = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NoUTurnSampler(new NoUTurnSamplerOptions { Chains = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NoUTurnSampler(new NoUTurnSamplerOptions { Draws = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NoUTurnSampler(new NoUTurnSamplerOptions { Warmup = -1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NoUTurnSampler(new NoUTurnSamplerOptions { Thin = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NoUTurnSampler(new NoUTurnSamplerOptions { MaxDegreeOfParallelism = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NoUTurnSampler(new NoUTurnSamplerOptions { Metric = (HamiltonianMetric)7 }));
        Assert.Throws<ArgumentException>(() => new NoUTurnSampler(new NoUTurnSamplerOptions { InitialInverseMetric = new Matrix<double>(2, 3) }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NoUTurnSampler(new NoUTurnSamplerOptions { Windows = new WarmupWindowOptions { BaseWindow = 0 } }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new HamiltonianMonteCarlo(new HamiltonianMonteCarloOptions { Steps = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new HamiltonianMonteCarlo(new HamiltonianMonteCarloOptions { StepSizeJitter = 1 }));
    }

    [Fact]
    public void OmittedSeedIsRecorded()
    {
        var target = new TargetDensities.IndependentNormal([0.0], [1.0]);
        HamiltonianSamplingResult first = new NoUTurnSampler(new NoUTurnSamplerOptions { Warmup = 50, Draws = 50 }).Sample(target, [0.0]);
        HamiltonianSamplingResult again = new NoUTurnSampler(new NoUTurnSamplerOptions { Warmup = 50, Draws = 50, Seed = first.Seed }).Sample(target, [0.0]);
        McmcAssert.Identical(first, again);
    }
}
