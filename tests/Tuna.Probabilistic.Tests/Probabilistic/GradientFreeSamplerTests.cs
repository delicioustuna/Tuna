using Tuna.LinearAlgebra;
using Tuna.Probabilistic.Inference;

namespace Tuna.Tests.Probabilistic;

/// <summary>勾配を使わないサンプラー（ランダムウォーク・メトロポリス、スライス、アンサンブル、楕円スライス）</summary>
[Trait("Probabilistic", "Mcmc")]
public class GradientFreeSamplerTests
{
    private static readonly TargetDensities.CorrelatedNormal Correlated = new(1, -2, 1, 10, 0.9);

    // ---------- ランダムウォーク・メトロポリス法 ----------

    [Fact]
    public void RandomWalkRecoversCorrelatedNormal()
    {
        RandomWalkMetropolisResult result = new RandomWalkMetropolis(new RandomWalkMetropolisOptions { Seed = 31, Draws = 5000 })
            .Sample(Correlated, [5.0, 5.0]);
        McmcAssert.Moments(result, [1, -2], Correlated.Covariance);
        Assert.All(result.AcceptanceRates, a => Assert.InRange(a, 0.12, 0.4));

        // 提案の共分散 s²C は目標の共分散に比例する（相関 0.9 を写し取る）
        for (int c = 0; c < result.ChainCount; c++)
        {
            double correlation = result.ProposalCovariances[0, 1, c] / Math.Sqrt(result.ProposalCovariances[0, 0, c] * result.ProposalCovariances[1, 1, c]);
            Assert.InRange(correlation, 0.75, 0.97);
        }
    }

    [Fact]
    public void RandomWalkTargetsOptimalAcceptanceInOneDimension()
    {
        RandomWalkMetropolisResult result = new RandomWalkMetropolis(new RandomWalkMetropolisOptions { Seed = 32, Draws = 5000 })
            .Sample(new TargetDensities.IndependentNormal([3.0], [0.2]), [0.0]);
        McmcAssert.Moments(result, [3], [0.04]);
        Assert.All(result.AcceptanceRates, a => Assert.InRange(a, 0.3, 0.6));
    }

    [Fact]
    public void RandomWalkHandlesDiscontinuousDensities()
    {
        RandomWalkMetropolisResult result = new RandomWalkMetropolis(new RandomWalkMetropolisOptions { Seed = 33, Draws = 5000 })
            .Sample(new TargetDensities.UnitBox(), [0.5, 0.5, 0.5]);
        McmcAssert.Moments(result, [0.5, 0.5, 0.5], [1.0 / 12, 1.0 / 12, 1.0 / 12]);
        Assert.All(result.Draws.AsSpan().ToArray(), v => Assert.InRange(v, 0, 1));
    }

    [Fact]
    public void RandomWalkCanReuseTunedProposal()
    {
        RandomWalkMetropolisResult tuned = new RandomWalkMetropolis(new RandomWalkMetropolisOptions { Seed = 34, Draws = 100 })
            .Sample(Correlated, [1.0, -2.0]);
        var proposal = new Matrix<double>(tuned.ProposalCovariances.GetMatrix(0).ToArray());
        var options = new RandomWalkMetropolisOptions { Seed = 35, Warmup = 0, Draws = 5000, InitialCovariance = proposal, ProposalScale = 1 };
        RandomWalkMetropolisResult reused = new RandomWalkMetropolis(options).Sample(Correlated, [1.0, -2.0]);
        Assert.All(reused.ProposalScales, s => Assert.Equal(1, s));
        McmcAssert.Moments(reused, [1, -2], Correlated.Covariance);

        // 適応させない場合は初期値のまま
        var fixedShape = new RandomWalkMetropolisOptions { Seed = 36, AdaptCovariance = false, Draws = 50, Warmup = 200 };
        RandomWalkMetropolisResult unadapted = new RandomWalkMetropolis(fixedShape).Sample(Correlated, [1.0, -2.0]);
        double s2 = unadapted.ProposalScales[0] * unadapted.ProposalScales[0];
        Assert.Equal(s2, unadapted.ProposalCovariances[0, 0, 0], 12);
        Assert.Equal(0, unadapted.ProposalCovariances[0, 1, 0]);
    }

    [Fact]
    public void RandomWalkIsReproducible()
    {
        var target = new TargetDensities.Banana(2, 0.03);
        RandomWalkMetropolisResult a = new RandomWalkMetropolis(new RandomWalkMetropolisOptions { Seed = 37, Draws = 300 }).Sample(target, [0.0, 0.0]);
        RandomWalkMetropolisResult b = new RandomWalkMetropolis(new RandomWalkMetropolisOptions { Seed = 37, Draws = 300, MaxDegreeOfParallelism = 3 })
            .Sample(x => target.LogDensity(x), [0.0, 0.0]);
        McmcAssert.Identical(a, b);
        Assert.True(a.AcceptanceRates.SequenceEqual(b.AcceptanceRates));

        // 評価の回数は、初期点 1 回と遷移ごとに 1 回
        Assert.All(a.Evaluations, e => Assert.Equal(1 + 1000 + 300, e));
    }

    [Fact]
    public void RandomWalkValidatesOptions()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new RandomWalkMetropolis(new RandomWalkMetropolisOptions { TargetAcceptance = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new RandomWalkMetropolis(new RandomWalkMetropolisOptions { ProposalScale = double.PositiveInfinity }));
        Assert.Throws<ArgumentException>(() => new RandomWalkMetropolis(new RandomWalkMetropolisOptions { InitialCovariance = new Matrix<double>(1, 2) }));
        var notPositive = new RandomWalkMetropolis(new RandomWalkMetropolisOptions { InitialCovariance = new Matrix<double>(2, 2) });
        Assert.Throws<ArgumentException>(() => notPositive.Sample(Correlated, [0.0, 0.0]));
        var wrongSize = new RandomWalkMetropolis(new RandomWalkMetropolisOptions { InitialCovariance = Matrix<double>.Identity(3) });
        Assert.Throws<ArgumentException>(() => wrongSize.Sample(Correlated, [0.0, 0.0]));
        Assert.Throws<ArgumentException>(() => new RandomWalkMetropolis().Sample(new TargetDensities.UnitBox(), [2.0]));
    }

    // ---------- スライス・サンプリング ----------

    [Fact]
    public void SliceRecoversCorrelatedNormal()
    {
        SliceSamplingResult result = new SliceSampler(new SliceSamplerOptions { Seed = 41 }).Sample(Correlated, [5.0, 5.0]);
        McmcAssert.Moments(result, [1, -2], Correlated.Covariance);

        // 方向は正規直交で、幅は各方向の標準偏差の 3 倍（目標の共分散の固有値から）
        var eigen = SymmetricEigenDecomposition.Compute(Correlated.Covariance);
        double[] expectedWidths = eigen.Values.Select(v => 3 * Math.Sqrt(v)).Order().ToArray();
        for (int c = 0; c < result.ChainCount; c++)
        {
            double dot = result.Directions[0, 0, c] * result.Directions[0, 1, c] + result.Directions[1, 0, c] * result.Directions[1, 1, c];
            Assert.Equal(0, dot, 12);
            double[] widths = [result.Widths[0, c], result.Widths[1, c]];
            Array.Sort(widths);
            Assert.InRange(widths[0] / expectedWidths[0], 0.7, 1.4);
            Assert.InRange(widths[1] / expectedWidths[1], 0.7, 1.4);
        }
    }

    [Fact]
    public void SliceWorksAlongCoordinateAxes()
    {
        var options = new SliceSamplerOptions { Seed = 42, AdaptDirections = false, Draws = 3000 };
        SliceSamplingResult result = new SliceSampler(options).Sample(new TargetDensities.IndependentNormal([0, 5], [0.01, 100]), [0.0, 0.0]);
        McmcAssert.Moments(result, [0, 5], [1e-4, 1e4]);
        Assert.Equal(1, result.Directions[0, 0, 0]);
        Assert.Equal(0, result.Directions[1, 0, 0]);
        Assert.InRange(result.Widths[1, 0] / 300, 0.7, 1.4);
    }

    [Fact]
    public void SliceHandlesDiscontinuousDensities()
    {
        SliceSamplingResult result = new SliceSampler(new SliceSamplerOptions { Seed = 43, AdaptWidths = false, AdaptDirections = false, InitialWidth = 0.3 })
            .Sample(new TargetDensities.UnitBox(), [0.2, 0.7]);
        McmcAssert.Moments(result, [0.5, 0.5], [1.0 / 12, 1.0 / 12]);
        Assert.All(result.Widths.AsSpan().ToArray(), w => Assert.Equal(0.3, w));
    }

    [Fact]
    public void SliceIsReproducibleAndValidated()
    {
        SliceSamplingResult a = new SliceSampler(new SliceSamplerOptions { Seed = 44, Draws = 200 }).Sample(Correlated, [0.0, 0.0]);
        SliceSamplingResult b = new SliceSampler(new SliceSamplerOptions { Seed = 44, Draws = 200, MaxDegreeOfParallelism = 4 })
            .Sample(x => Correlated.LogDensity(x), [0.0, 0.0]);
        McmcAssert.Identical(a, b);
        Assert.Throws<ArgumentOutOfRangeException>(() => new SliceSampler(new SliceSamplerOptions { InitialWidth = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SliceSampler(new SliceSamplerOptions { MaxSteppingOut = 0 }));
        Assert.Throws<ArgumentNullException>(() => new SliceSampler(new SliceSamplerOptions { Windows = null! }));
    }

    // ---------- アンサンブル・サンプラー ----------

    [Fact]
    public void EnsembleRecoversCorrelatedNormal()
    {
        EnsembleSamplingResult result = new EnsembleSampler(new EnsembleSamplerOptions { Seed = 51, Draws = 3000 }).Sample(Correlated, [0.0, 0.0]);
        Assert.Equal(8, result.ChainCount);
        McmcAssert.Moments(result, [1, -2], Correlated.Covariance);
        Assert.InRange(result.MeanAcceptanceRate, 0.4, 0.9);

        // アフィン不変: 座標を線形変換した目標でも受理率は同じ程度
        var stretched = new TargetDensities.CorrelatedNormal(0, 0, 1e-3, 1e3, 0.99);
        EnsembleSamplingResult other = new EnsembleSampler(new EnsembleSamplerOptions { Seed = 51, Draws = 3000 }).Sample(stretched, [0.0, 0.0]);
        Assert.InRange(other.MeanAcceptanceRate, 0.4, 0.9);
        McmcAssert.Moments(other, [0, 0], stretched.Covariance);
    }

    [Fact]
    public void EnsembleIsIndependentOfParallelism()
    {
        var target = new TargetDensities.Banana(3, 0.02);
        var options = new EnsembleSamplerOptions { Seed = 52, Walkers = 10, Draws = 200, Warmup = 100 };
        EnsembleSamplingResult sequential = new EnsembleSampler(options).Sample(target, [0.0, 0.0, 0.0]);
        EnsembleSamplingResult parallel = new EnsembleSampler(new EnsembleSamplerOptions { Seed = 52, Walkers = 10, Draws = 200, Warmup = 100, MaxDegreeOfParallelism = -1 })
            .Sample(x => target.LogDensity(x), [0.0, 0.0, 0.0]);
        McmcAssert.Identical(sequential, parallel);
        Assert.True(sequential.AcceptanceRates.SequenceEqual(parallel.AcceptanceRates));
        Assert.Equal(10, sequential.ChainCount);
    }

    [Fact]
    public void EnsembleAcceptsWalkerPositions()
    {
        var starts = new Matrix<double>(2, 6);
        for (int w = 0; w < 6; w++)
        {
            starts[0, w] = Math.Cos(w);
            starts[1, w] = Math.Sin(w) * 5;
        }

        EnsembleSamplingResult result = new EnsembleSampler(new EnsembleSamplerOptions { Seed = 53, Draws = 4000 }).Sample(Correlated, starts);
        Assert.Equal(6, result.ChainCount);
        McmcAssert.Moments(result, [1, -2], Correlated.Covariance);

        Assert.Throws<ArgumentException>(() => new EnsembleSampler(new EnsembleSamplerOptions { Walkers = 4 }).Sample(Correlated, starts));
        Assert.Throws<ArgumentException>(() => new EnsembleSampler(new EnsembleSamplerOptions { Walkers = 3 }).Sample(Correlated, new double[3]));
        Assert.Throws<ArgumentException>(() => new EnsembleSampler().Sample(new TargetDensities.UnitBox(), [5.0]));
        Assert.Throws<ArgumentOutOfRangeException>(() => new EnsembleSampler(new EnsembleSamplerOptions { Walkers = 1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new EnsembleSampler(new EnsembleSamplerOptions { StretchScale = 1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new EnsembleSampler(new EnsembleSamplerOptions { InitialSpread = 0 }));
    }

    // ---------- 楕円スライス・サンプリング ----------

    /// <summary>事前分布 N(0, Σ)、尤度 N(y; θ, I) の事後分布 N((Σ⁻¹ + I)⁻¹y, (Σ⁻¹ + I)⁻¹)</summary>
    private static (Matrix<double> Prior, double[] Y, double[] Mean, Matrix<double> Covariance) GaussianPosterior()
    {
        var prior = new Matrix<double>(new[,] { { 4.0, 1.8 }, { 1.8, 1.0 } });
        double[] y = [1.0, 2.0];
        double det = 4 - 1.8 * 1.8;
        var precision = new Matrix<double>(new[,] { { 1.0 / det + 1, -1.8 / det }, { -1.8 / det, 4.0 / det + 1 } });
        double pd = precision[0, 0] * precision[1, 1] - precision[0, 1] * precision[1, 0];
        var covariance = new Matrix<double>(new[,] { { precision[1, 1] / pd, -precision[0, 1] / pd }, { -precision[1, 0] / pd, precision[0, 0] / pd } });
        double[] mean = [covariance[0, 0] * y[0] + covariance[0, 1] * y[1], covariance[1, 0] * y[0] + covariance[1, 1] * y[1]];
        return (prior, y, mean, covariance);
    }

    [Fact]
    public void EllipticalSliceRecoversGaussianPosterior()
    {
        var (prior, y, mean, covariance) = GaussianPosterior();
        double LogLikelihood(ReadOnlyVectorSpan<double> x) => -0.5 * ((x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]));
        EllipticalSliceSamplingResult result = new EllipticalSliceSampler(new EllipticalSliceSamplerOptions { Seed = 61 })
            .Sample(LogLikelihood, [0.0, 0.0], prior);
        McmcAssert.Moments(result, mean, covariance);

        // 対数密度は対数尤度と −½(θ − μ)ᵀΣ⁻¹(θ − μ) の和
        for (int t = 0; t < 5; t++)
        {
            double a = result.Draws[0, t, 1], b = result.Draws[1, t, 1];
            double det = 4 - 1.8 * 1.8;
            double quadratic = (a * a * 1.0 - 2 * 1.8 * a * b + 4.0 * b * b) / det;
            Assert.Equal(LogLikelihood(new[] { a, b }), result.LogLikelihoods[t, 1], 12);
            Assert.Equal(result.LogLikelihoods[t, 1] - 0.5 * quadratic, result.LogDensities[t, 1], 10);
        }
    }

    [Fact]
    public void EllipticalSliceHandlesBoundedLikelihoods()
    {
        // 尤度が台 [0, 1]² の指示関数なら、事後分布は切断した事前分布 N(0, I)。切断正規分布 N(0, 1) を [0, 1] に切った平均・分散
        var identity = Matrix<double>.Identity(2);
        var starts = new Matrix<double>(2, 4);
        starts.AsSpan().Fill(0.5);
        EllipticalSliceSamplingResult result = new EllipticalSliceSampler(new EllipticalSliceSamplerOptions { Seed = 62 })
            .Sample(new TargetDensities.UnitBox(), [0.0, 0.0], identity, starts);
        double z = XMath.Special.NormalCdf(1) - 0.5;
        double phi0 = 1 / Math.Sqrt(2 * Math.PI), phi1 = phi0 * Math.Exp(-0.5);
        double mean = (phi0 - phi1) / z;
        double variance = 1 - phi1 / z - mean * mean;
        McmcAssert.Moments(result, [mean, mean], [variance, variance]);
    }

    [Fact]
    public void EllipticalSliceIsReproducibleAndValidated()
    {
        var (prior, y, _, _) = GaussianPosterior();
        double LogLikelihood(ReadOnlyVectorSpan<double> x) => -0.5 * ((x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]));
        EllipticalSliceSamplingResult a = new EllipticalSliceSampler(new EllipticalSliceSamplerOptions { Seed = 63, Draws = 100 })
            .Sample(LogLikelihood, [0.0, 0.0], prior);
        EllipticalSliceSamplingResult b = new EllipticalSliceSampler(new EllipticalSliceSamplerOptions { Seed = 63, Draws = 100, MaxDegreeOfParallelism = 4 })
            .Sample(LogLikelihood, [0.0, 0.0], prior);
        McmcAssert.Identical(a, b);

        var sampler = new EllipticalSliceSampler();
        Assert.Throws<ArgumentException>(() => sampler.Sample(LogLikelihood, ReadOnlySpan<double>.Empty, prior));
        Assert.Throws<ArgumentException>(() => sampler.Sample(LogLikelihood, [0.0, double.NaN], prior));
        Assert.Throws<ArgumentException>(() => sampler.Sample(LogLikelihood, [0.0, 0.0, 0.0], prior));
        Assert.Throws<ArgumentException>(() => sampler.Sample(LogLikelihood, [0.0, 0.0], new Matrix<double>(new[,] { { 1.0, 2.0 }, { 2.0, 1.0 } })));
        Assert.Throws<ArgumentException>(() => sampler.Sample(x => double.NegativeInfinity, [0.0, 0.0], prior));
        Assert.Throws<ArgumentException>(() => sampler.Sample(LogLikelihood, [0.0, 0.0], prior, new Matrix<double>(2, 3)));
        Assert.Throws<ArgumentException>(() => sampler.Sample(LogLikelihood, [0.0, 0.0], prior, new Matrix<double>(3, 4)));
    }
}
