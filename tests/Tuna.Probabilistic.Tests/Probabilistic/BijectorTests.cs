using Tuna.LinearAlgebra;
using Tuna.Probabilistic.Inference;

namespace Tuna.Tests.Probabilistic;

/// <summary>制約変換: 往復、ヤコビアンの対数行列式と勾配の引き戻しを差分と比べる、端の数値の扱い、制約のある分布からの標本</summary>
[Trait("Probabilistic", "Mcmc")]
public class BijectorTests
{
    public static TheoryData<string> Bijectors() => new(Catalog.Keys);

    private static readonly Dictionary<string, IBijector> Catalog = new()
    {
        ["Interval"] = BoxBijector.Uniform(3, -2, 5),
        ["Positive"] = BoxBijector.Positive(2),
        ["Mixed"] = new BoxBijector([0, double.NegativeInfinity, double.NegativeInfinity, 1e-3], [double.PositiveInfinity, 4, double.PositiveInfinity, 2e-3]),
        ["Simplex2"] = new SimplexBijector(2),
        ["Simplex5"] = new SimplexBijector(5),
        ["Ordered"] = new OrderedBijector(4),
        ["Composite"] = new CompositeBijector(BoxBijector.Positive(1), new SimplexBijector(3), new OrderedBijector(2)),
    };

    private static double[] Point(int n, int seed)
    {
        var random = new Tuna.RandomSource.Philox4x32Random((ulong)seed);
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = 3 * (random.NextDouble() - 0.5);
        }

        return y;
    }

    /// <summary>制約のない点の写像 y ↦ （先頭 n 成分の）x</summary>
    private static double[] Forward(IBijector bijector, double[] y)
    {
        var x = new double[bijector.ConstrainedDimension];
        bijector.Forward(y, x);
        return x;
    }

    [Theory]
    [MemberData(nameof(Bijectors))]
    public void InverseUndoesForward(string name)
    {
        IBijector bijector = Catalog[name];
        for (int seed = 0; seed < 20; seed++)
        {
            double[] y = Point(bijector.UnconstrainedDimension, seed);
            double[] x = Forward(bijector, y);
            var back = new double[y.Length];
            bijector.Inverse(x, back);
            for (int i = 0; i < y.Length; i++)
            {
                Assert.Equal(y[i], back[i], 1e-12 * (1 + Math.Abs(y[i])));
            }
        }
    }

    [Theory]
    [MemberData(nameof(Bijectors))]
    public void LogJacobianMatchesFiniteDifferences(string name)
    {
        // 単体では密度を先頭 K − 1 成分についてのものとするので、その成分のヤコビアンの行列式と比べる（変換の制約のない次元 n の正方行列）
        IBijector bijector = Catalog[name];
        int n = bijector.UnconstrainedDimension;
        for (int seed = 0; seed < 5; seed++)
        {
            double[] y = Point(n, 100 + seed);
            double logJacobian = bijector.Forward(y, new double[bijector.ConstrainedDimension]);
            var jacobian = new Matrix<double>(n, n);
            int[] rows = DensityCoordinates(bijector);
            for (int j = 0; j < n; j++)
            {
                double h = 1e-6 * (1 + Math.Abs(y[j]));
                double[] plus = (double[])y.Clone(), minus = (double[])y.Clone();
                plus[j] += h;
                minus[j] -= h;
                double[] xp = Forward(bijector, plus), xm = Forward(bijector, minus);
                for (int i = 0; i < n; i++)
                {
                    jacobian[i, j] = (xp[rows[i]] - xm[rows[i]]) / (2 * h);
                }
            }

            double expected = LuDecomposition.Compute(jacobian).LogAbsDeterminant;
            Assert.Equal(expected, logJacobian, 1e-6 * (1 + Math.Abs(expected)));
        }
    }

    /// <summary>密度の座標として使う制約のある成分（単体では各区画の最後の成分を除く）</summary>
    private static int[] DensityCoordinates(IBijector bijector) => bijector switch
    {
        SimplexBijector s => Enumerable.Range(0, s.UnconstrainedDimension).ToArray(),
        CompositeBijector => [0, 1, 2, 4, 5],
        _ => Enumerable.Range(0, bijector.UnconstrainedDimension).ToArray(),
    };

    /// <summary>制約のある空間の滑らかな試験関数 h(x) = Σ cₖxₖ − ½Σ xₖ²</summary>
    private readonly struct Quadratic(int n) : IDifferentiableLogDensity
    {
        public double LogDensity(ReadOnlyVectorSpan<double> x, VectorSpan<double> g)
        {
            double sum = 0;
            for (int k = 0; k < n; k++)
            {
                double c = 0.3 * (k + 1);
                sum += c * x[k] - 0.5 * x[k] * x[k];
                g[k] = c - x[k];
            }

            return sum;
        }
    }

    [Theory]
    [MemberData(nameof(Bijectors))]
    public void PullBackMatchesFiniteDifferences(string name)
    {
        IBijector bijector = Catalog[name];
        var density = new TransformedDifferentiableLogDensity<Quadratic, IBijector>(new Quadratic(bijector.ConstrainedDimension), bijector);
        for (int seed = 0; seed < 5; seed++)
        {
            double[] y = Point(bijector.UnconstrainedDimension, 200 + seed);
            Assert.True(LogDensities.GradientError(density, y) < 1e-7, name);

            // 値だけの評価と勾配つきの評価は同じ値
            var gradient = new double[y.Length];
            Assert.Equal(density.LogDensity(y), density.LogDensity(y, gradient));
            var valueOnly = new TransformedLogDensity<Quadratic, IBijector>(new Quadratic(bijector.ConstrainedDimension), bijector);
            Assert.Equal(valueOnly.LogDensity(y), density.LogDensity(y));
        }
    }

    [Fact]
    public void ExtremeValuesStayInsideTheConstraints()
    {
        // 区間の上限の近くは上限からの差として求めるので、上限 0 の近くの小さい値が 0 に丸められない
        var interval = BoxBijector.Uniform(1, -1, 0);
        var x = new double[1];
        interval.Forward(new[] { 40.0 }, x);
        Assert.True(x[0] < 0 && x[0] > -1e-17);
        interval.Forward(new[] { -30.0 }, x);
        Assert.True(x[0] > -1 && x[0] < -1 + 1e-12);

        // 単体の非常に小さい成分も正のまま、和は 1
        var simplex = new SimplexBijector(4);
        var w = new double[4];
        simplex.Forward(new[] { -300.0, 0.0, 300.0 }, w);
        Assert.True(w[0] > 0 && w[0] < 1e-120);
        Assert.Equal(1, w.Sum(), 15);

        // 境界上の点は ±∞、外の点は NaN
        var y = new double[1];
        interval.Inverse(new[] { 0.0 }, y);
        Assert.Equal(double.PositiveInfinity, y[0]);
        interval.Inverse(new[] { -1.0 }, y);
        Assert.Equal(double.NegativeInfinity, y[0]);
        interval.Inverse(new[] { 0.5 }, y);
        Assert.True(double.IsNaN(y[0]));
        var v = new double[3];
        simplex.Inverse(new[] { 0.5, 0.5, 0.1, 0.0 }, v);
        Assert.All(v, e => Assert.True(double.IsNaN(e)));
        simplex.Inverse(new[] { 0.5, -0.1, 0.3, 0.3 }, v);
        Assert.All(v, e => Assert.True(double.IsNaN(e)));
        var ordered = new OrderedBijector(3);
        ordered.Inverse(new[] { 0.0, 1.0, 1.0 }, v);
        Assert.Equal(double.NegativeInfinity, v[2]);
        ordered.Inverse(new[] { 0.0, 1.0, 0.5 }, v);
        Assert.True(double.IsNaN(v[2]));
    }

    [Fact]
    public void ArgumentsAreValidated()
    {
        Assert.Throws<ArgumentException>(() => new BoxBijector([0.0], [0.0]));
        Assert.Throws<ArgumentException>(() => new BoxBijector([0.0], [1.0, 2.0]));
        Assert.Throws<ArgumentException>(() => new BoxBijector([], []));
        Assert.Throws<ArgumentException>(() => new BoxBijector([double.NaN], [1.0]));
        Assert.Throws<ArgumentException>(() => new BoxBijector([double.PositiveInfinity], [double.PositiveInfinity]));
        Assert.Throws<ArgumentException>(() => new BoxBijector([-double.MaxValue], [double.MaxValue]));
        Assert.Throws<ArgumentOutOfRangeException>(() => BoxBijector.Positive(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SimplexBijector(1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new OrderedBijector(0));
        Assert.Throws<ArgumentException>(() => new CompositeBijector());
        Assert.Throws<ArgumentException>(() => new CompositeBijector(new OrderedBijector(1), null!));
        Assert.Throws<ArgumentException>(() => new SimplexBijector(3).Forward(new double[3], new double[3]));
        Assert.Throws<ArgumentException>(() => new OrderedBijector(2).PullBack(new double[2], new double[2], new double[1]));
        Assert.Throws<ArgumentNullException>(() => new TransformedLogDensity<Quadratic, IBijector>(new Quadratic(1), null!));

        var box = new BoxBijector([0.0, 1.0], [1.0, 2.0]);
        double[] lower = box.LowerBounds;
        lower[0] = 99;
        Assert.Equal(0, box.LowerBounds[0]);
        Assert.Equal(new[] { 1.0, 2.0 }, box.UpperBounds);
    }

    [Fact]
    public void SamplesBetaThroughTheIntervalTransform()
    {
        var density = new TransformedDifferentiableLogDensity<TargetDensities.Beta, BoxBijector>(new TargetDensities.Beta(2, 3), BoxBijector.Uniform(1, 0, 1));
        HamiltonianSamplingResult result = new NoUTurnSampler(new NoUTurnSamplerOptions { Seed = 81 }).Sample(density, [0.0]);
        SamplingResult constrained = result.Transform(BoxBijector.Uniform(1, 0, 1));
        Assert.All(constrained.Draws.AsSpan().ToArray(), v => Assert.InRange(v, 0, 1));
        McmcAssert.Moments(constrained, [0.4], [0.04]);
        Assert.True(constrained.LogDensities.AsSpan().SequenceEqual(result.LogDensities.AsSpan()));
    }

    [Fact]
    public void SamplesDirichletThroughTheSimplexTransform()
    {
        var simplex = new SimplexBijector(3);
        var density = new TransformedDifferentiableLogDensity<TargetDensities.Dirichlet, SimplexBijector>(new TargetDensities.Dirichlet([1, 2, 3]), simplex);
        HamiltonianSamplingResult result = new NoUTurnSampler(new NoUTurnSamplerOptions { Seed = 82 }).Sample(density, [0.0, 0.0]);
        SamplingResult constrained = result.Transform(simplex);
        Assert.Equal(3, constrained.Dimension);
        McmcAssert.Moments(constrained, [1.0 / 6, 2.0 / 6, 3.0 / 6], [5.0 / 252, 8.0 / 252, 9.0 / 252]);
        McmcAssert.Expectation(constrained, x => x[0] + x[1] + x[2], 1, "sum", minimumEss: 0);
    }

    [Fact]
    public void SamplesOrderedPairs()
    {
        // 独立な2つの標準正規を x₁ < x₂ に制限した分布: E[x₁] = −1/√π、E[x₂] = 1/√π、分散 1 − 1/π
        var ordered = new OrderedBijector(2);
        var density = new TransformedLogDensity<TargetDensities.IndependentNormal, OrderedBijector>(new TargetDensities.IndependentNormal([0, 0], [1, 1]), ordered);
        SliceSamplingResult result = new SliceSampler(new SliceSamplerOptions { Seed = 83, Draws = 3000 }).Sample(density, [0.0, 0.0]);
        SamplingResult constrained = result.Transform(ordered);
        double m = 1 / Math.Sqrt(Math.PI);
        McmcAssert.Moments(constrained, [-m, m], [1 - 1 / Math.PI, 1 - 1 / Math.PI]);
        Assert.Throws<ArgumentException>(() => result.Transform(new OrderedBijector(3)));
    }
}
