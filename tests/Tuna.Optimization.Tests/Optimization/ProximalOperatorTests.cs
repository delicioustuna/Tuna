using Tuna.LinearAlgebra;
using Tuna.Optimization;
using Tuna.Optimization.Proximal;
using static Tuna.Tests.Optimization.GradientTestFunctions;

namespace Tuna.Tests.Optimization;

[Trait("Optimization", "Proximal")]
public class ProximalOperatorTests
{
    // ---------- 近接写像の最適性 ----------

    public static TheoryData<string> OperatorNames => new(Operators(7).Select(o => o.Name));

    private static IEnumerable<(string Name, IProximalOperator Operator)> Operators(int n)
    {
        var random = new Random(5);
        var weights = Enumerable.Range(0, n).Select(_ => random.NextDouble() * 2).ToArray();
        var lower = Enumerable.Range(0, n).Select(i => i % 3 == 0 ? double.NegativeInfinity : -0.5).ToArray();
        var upper = Enumerable.Range(0, n).Select(i => i % 4 == 0 ? double.PositiveInfinity : 0.7).ToArray();
        yield return ("Zero", new ZeroFunction());
        yield return ("L1", new L1Penalty(0.4));
        yield return ("WeightedL1", new L1Penalty(0.4, weights));
        yield return ("ElasticNet", new ElasticNetPenalty(0.3, 0.8));
        yield return ("L2Norm", new L2NormPenalty(0.9));
        yield return ("GroupLasso", new GroupLassoPenalty(0.5, [2, 1, 4]));
        yield return ("NonNegative", new NonNegativeConstraint());
        yield return ("Box", new BoxConstraint(lower, upper));
        yield return ("L2Ball", new L2BallConstraint(0.8));
        yield return ("L1Ball", new L1BallConstraint(1.1));
        yield return ("Simplex", new SimplexConstraint(1.5));
        yield return ("LeastSquaresTall", new LeastSquaresObjective(RandomMatrix(9, n, 1), RandomVector(9, 2)));
        yield return ("LeastSquaresWide", new LeastSquaresObjective(RandomMatrix(4, n, 3), RandomVector(4, 4)));
        yield return ("Quadratic", new QuadraticObjective(RandomPositiveDefinite(n, 0.0, 6), RandomVector(n, 7), 1.5));
    }

    [Theory]
    [MemberData(nameof(OperatorNames))]
    public void ProximalPointMinimizesTheProximalObjective(string name)
    {
        // p = prox_{t·h}(v) は φ(x) = h(x) + ‖x − v‖²/(2t) の最小点。定義域の中の点との凸結合と、小さな摂動で値が下がらないことを確かめる
        const int n = 7;
        var h = Operators(n).Single(o => o.Name == name).Operator;
        var random = new Random(name.GetHashCode() & 0xffff);
        foreach (double t in new[] { 0.05, 0.7, 3.0 })
        {
            for (int trial = 0; trial < 20; trial++)
            {
                var v = RandomPoint(random, n, scale: 2);
                var p = new double[n];
                h.Apply(v, t, p);
                double best = Phi(h, p, v, t);
                Assert.True(double.IsFinite(best), $"{name}: φ(prox) = {best}");
                double tolerance = 1e-12 * (1 + Math.Abs(best));
                for (int k = 0; k < 10; k++)
                {
                    var w = new double[n];
                    h.Apply(RandomPoint(random, n, scale: 3), t, w);
                    foreach (double lambda in new[] { 1e-3, 0.1, 0.5, 1.0 })
                    {
                        var q = p.Zip(w, (a, b) => (1 - lambda) * a + lambda * b).ToArray();
                        Assert.True(Phi(h, q, v, t) >= best - tolerance, $"{name}: convex combination {lambda} beats the prox");
                    }

                    var nudged = p.Select(a => a + 1e-4 * (2 * random.NextDouble() - 1)).ToArray();
                    Assert.True(Phi(h, nudged, v, t) >= best - tolerance, $"{name}: perturbation beats the prox");
                }
            }
        }
    }

    [Theory]
    [MemberData(nameof(OperatorNames))]
    public void ApplyWorksInPlaceAndOnStridedViews(string name)
    {
        const int n = 7;
        var h = Operators(n).Single(o => o.Name == name).Operator;
        var v = RandomPoint(new Random(9), n, scale: 2);
        var expected = new double[n];
        h.Apply(v, 0.6, expected);

        var inPlace = (double[])v.Clone();
        h.Apply(inPlace, 0.6, inPlace);
        Assert.Equal(expected, inPlace);

        // 1つおきに並べたビュー（SIMD を使わない経路）
        var source = new double[2 * n];
        var target = new double[2 * n];
        for (int i = 0; i < n; i++)
        {
            source[2 * i] = v[i];
            source[2 * i + 1] = double.NaN;
            target[2 * i + 1] = 42;
        }

        h.Apply(new ReadOnlyVectorSpan<double>(source, [(nint)n], [(nint)2]), 0.6, new VectorSpan<double>(target, [(nint)n], [(nint)2]));
        for (int i = 0; i < n; i++)
        {
            Assert.Equal(expected[i], target[2 * i], 1e-14);
            Assert.Equal(42.0, target[2 * i + 1]);
        }
    }

    // ---------- 閉じた形・恒等式 ----------

    [Fact]
    public void SoftThresholdMatchesTheFormulaAcrossSimdWidths()
    {
        // 長さ 37 で SIMD の本体と端の両方を通す
        var v = RandomPoint(new Random(12), 37, scale: 3);
        var x = new double[37];
        new L1Penalty(0.5).Apply(v, 2.0, x);
        for (int i = 0; i < v.Length; i++)
        {
            Assert.Equal(Math.Sign(v[i]) * Math.Max(Math.Abs(v[i]) - 1.0, 0), x[i], 1e-15);
        }

        new ElasticNetPenalty(0.5, 3.0).Apply(v, 2.0, x);
        for (int i = 0; i < v.Length; i++)
        {
            Assert.Equal(Math.Sign(v[i]) * Math.Max(Math.Abs(v[i]) - 1.0, 0) / 7, x[i], 1e-15);
        }

        Assert.Equal(0.5 * v.Sum(Math.Abs), new L1Penalty(0.5).Evaluate(v), 1e-12);
    }

    [Fact]
    public void MoreauDecompositionHoldsForNormsAndTheirDualBalls()
    {
        // v = prox_{t·h}(v) + t·proj_{h*}(v/t)。‖·‖₂ の双対は L2 球、‖·‖₁ の双対は箱 [−λ, λ]
        const int n = 11;
        const double lambda = 0.7, t = 1.3;
        var random = new Random(13);
        for (int trial = 0; trial < 20; trial++)
        {
            var v = RandomPoint(random, n, scale: trial % 2 == 0 ? 0.5 : 3);
            var scaled = v.Select(a => a / t).ToArray();
            var prox = new double[n];
            var proj = new double[n];

            new L2NormPenalty(lambda).Apply(v, t, prox);
            new L2BallConstraint(lambda).Apply(scaled, 1.0, proj);
            for (int i = 0; i < n; i++)
            {
                Assert.Equal(v[i], prox[i] + t * proj[i], 1e-13);
            }

            new L1Penalty(lambda).Apply(v, t, prox);
            new BoxConstraint(Enumerable.Repeat(-lambda, n).ToArray(), Enumerable.Repeat(lambda, n).ToArray()).Apply(scaled, 1.0, proj);
            for (int i = 0; i < n; i++)
            {
                Assert.Equal(v[i], prox[i] + t * proj[i], 1e-13);
            }
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(17)]
    [InlineData(1000)]
    public void SimplexProjectionMatchesTheSortingMethod(int n)
    {
        var random = new Random(n);
        var simplex = new SimplexConstraint(2.5);
        var x = new double[n];
        for (int trial = 0; trial < 30; trial++)
        {
            var v = RandomPoint(random, n, scale: trial % 3 == 0 ? 0.01 : 5);
            if (trial % 5 == 1)
            {
                // 同じ値が多い場合
                for (int i = 0; i < n; i++)
                {
                    v[i] = Math.Round(v[i]);
                }
            }

            simplex.Apply(v, 1.0, x);
            double tau = SortingThreshold(v, 2.5);
            for (int i = 0; i < n; i++)
            {
                Assert.Equal(Math.Max(v[i] - tau, 0), x[i], 1e-12);
            }

            Assert.Equal(2.5, x.Sum(), 1e-11);
            Assert.Equal(0.0, simplex.Evaluate(x));
        }

        Assert.Equal(double.PositiveInfinity, simplex.Evaluate(Enumerable.Repeat(1.0, n).ToArray()));
    }

    [Fact]
    public void L1BallProjectionShrinksOnlyOutsideTheBall()
    {
        var ball = new L1BallConstraint(2.0);
        double[] inside = [0.5, -1.0, 0.25];
        var x = new double[3];
        ball.Apply(inside, 1.0, x);
        Assert.Equal(inside, x);

        var random = new Random(14);
        for (int trial = 0; trial < 30; trial++)
        {
            var v = RandomPoint(random, 25, scale: 2);
            var p = new double[25];
            ball.Apply(v, 1.0, p);
            double tau = SortingThreshold(v.Select(Math.Abs).ToArray(), 2.0);
            for (int i = 0; i < 25; i++)
            {
                Assert.Equal(Math.Sign(v[i]) * Math.Max(Math.Abs(v[i]) - tau, 0), p[i], 1e-12);
            }

            Assert.Equal(2.0, p.Sum(Math.Abs), 1e-12);
        }
    }

    // 和・半径に比べて極端に大きな入力に対する単体・L1 球への射影を検証する。
    [Theory]
    [InlineData(0.0)]
    [InlineData(1e16)]
    [InlineData(-1e16)]
    [InlineData(1e300)]
    [InlineData(-1e300)]
    [InlineData(double.MaxValue)]
    public void SimplexProjectionIsInvariantUnderCommonTranslation(double offset)
    {
        var constraint = new SimplexConstraint();
        double[] output = new double[2];
        constraint.Apply(new double[] { offset, offset }, 1, output);
        Assert.Equal(new double[] { 0.5, 0.5 }, output);
        Assert.Equal(0.0, constraint.Evaluate(output));
    }

    [Fact]
    public void SimplexProjectionOfLargeTranslatedPointsMatchesTheUntranslatedProjection()
    {
        var constraint = new SimplexConstraint(1.5);
        var random = new Random(20261014);
        var expected = new double[5];
        var actual = new double[5];
        for (int trial = 0; trial < 50; trial++)
        {
            // 平行移動した点が正確に表現できるように、成分は [−2, 2] の 1/8 の倍数、平行移動は 2⁴⁴〜2⁴⁹（ulp ≤ 1/8）にする。
            // 以前は τ = m + (τ − m) が 1/8 単位に丸められ、この程度の大きさでも和が 1.5 からずれた
            var v = Enumerable.Range(0, 5).Select(_ => random.Next(-16, 17) / 8.0).ToArray();
            double offset = (trial % 2 == 0 ? 1 : -1) * Math.ScaleB(1, 44 + trial % 6);
            constraint.Apply(v, 1, expected);
            constraint.Apply(v.Select(x => x + offset).ToArray(), 1, actual);
            for (int i = 0; i < 5; i++)
            {
                Assert.Equal(expected[i], actual[i], 1e-15);
            }

            Assert.Equal(0.0, constraint.Evaluate(actual));
        }

        // 平行移動の差がすべての成分の差より大きく、ほとんどの成分が −total 以下になる場合（z の計算があふれる場合を含む）
        double[] extreme = new double[3];
        constraint.Apply(new double[] { -double.MaxValue, double.MaxValue, -1e300 }, 1, extreme);
        Assert.Equal(new double[] { 0.0, 1.5, 0.0 }, extreme);
    }

    [Theory]
    [InlineData(1e16, 1.0)]
    [InlineData(-1e16, -1.0)]
    [InlineData(1e300, 1.0)]
    public void L1BallProjectionOfOneLargeCoordinateEqualsTheRadius(double value, double expected)
    {
        double[] output = new double[1];
        new L1BallConstraint(1).Apply(new double[] { value }, 1, output);
        Assert.Equal(expected, output[0]);
    }

    [Fact]
    public void L1BallProjectionOfLargeEqualMagnitudesSharesTheRadius()
    {
        var ball = new L1BallConstraint(2);
        double[] output = new double[4];
        ball.Apply(new double[] { 1e16, -1e16, 1e16, -1e16 }, 1, output);
        Assert.Equal(new double[] { 0.5, -0.5, 0.5, -0.5 }, output);
        Assert.Equal(0.0, ball.Evaluate(output));
    }

    [Fact]
    public void LeastSquaresProximalMatchesADirectSolve()
    {
        foreach ((int m, int n) in new[] { (12, 5), (3, 8) })
        {
            var a = RandomMatrix(m, n, m);
            var b = RandomVector(m, n);
            var f = new LeastSquaresObjective(a, b);
            var v = RandomVector(n, 99);
            foreach (double t in new[] { 0.3, 2.0, 0.3 })   // 歩幅を戻しても分解を作り直して正しい
            {
                var x = new double[n];
                f.Apply(v, t, x);

                // (AᵀA + I/t)x = Aᵀb + v/t
                var lhs = new Matrix<double>(n, n);
                var rhs = new Vector<double>(n);
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        double sum = i == j ? 1 / t : 0;
                        for (int k = 0; k < m; k++)
                        {
                            sum += a[k, i] * a[k, j];
                        }

                        lhs[i, j] = sum;
                    }

                    double r = v[i] / t;
                    for (int k = 0; k < m; k++)
                    {
                        r += a[k, i] * b[k];
                    }

                    rhs[i] = r;
                }

                var expected = lhs.Cholesky().Solve(rhs);
                for (int i = 0; i < n; i++)
                {
                    Assert.Equal(expected[i], x[i], 1e-11);
                }
            }

            Assert.True(GradientCheck.MaximumError(f, RandomVector(n, 5).ToArray()) < 1e-8);
            var point = RandomVector(n, 6);
            double residual = 0;
            for (int k = 0; k < m; k++)
            {
                double r = -b[k];
                for (int j = 0; j < n; j++)
                {
                    r += a[k, j] * point[j];
                }

                residual += r * r;
            }

            Assert.Equal(0.5 * residual, f.Evaluate(point), 1e-12);
        }
    }

    [Fact]
    public void QuadraticObjectiveChecksItsInputs()
    {
        var p = RandomPositiveDefinite(4, 0.5, 1);
        var q = new QuadraticObjective(p, RandomVector(4, 2), constant: 3);
        Assert.Equal(3.0, q.Evaluate(new double[4]));
        Assert.True(GradientCheck.MaximumError(q, [0.3, -0.2, 1.0, 0.5]) < 1e-8);

        // 正定値でない P では近接写像が作れないことがある
        var indefinite = new Matrix<double>(2, 2);
        indefinite[0, 0] = 1;
        indefinite[1, 1] = -5;
        Assert.Throws<InvalidOperationException>(() => new QuadraticObjective(indefinite, new double[2]).Apply(new double[2], 1.0, new double[2]));
        Assert.Throws<ArgumentException>(() => new QuadraticObjective(new Matrix<double>(2, 3), new double[2]));
        Assert.Throws<ArgumentException>(() => new QuadraticObjective(p, new double[3]));
        Assert.Throws<ArgumentException>(() => new LeastSquaresObjective(new Matrix<double>(3, 2), new double[2]));
    }

    [Fact]
    public void RejectsInvalidArguments()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new L1Penalty(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new L1Penalty(1, [1.0, double.NaN]));
        Assert.Throws<ArgumentException>(() => new L1Penalty(1, []));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ElasticNetPenalty(1, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new GroupLassoPenalty(1, [2, 0]));
        Assert.Throws<ArgumentException>(() => new GroupLassoPenalty(1, []));
        Assert.Throws<ArgumentException>(() => new BoxConstraint([], []));
        Assert.Throws<ArgumentException>(() => new BoxConstraint([1.0], [0.0]));
        Assert.Throws<ArgumentOutOfRangeException>(() => new L1BallConstraint(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SimplexConstraint(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new L1Penalty(1).Apply(new double[2], 0, new double[2]));
        Assert.Throws<ArgumentException>(() => new L1Penalty(1).Apply(new double[2], 1, new double[3]));
        Assert.Throws<ArgumentException>(() => new GroupLassoPenalty(1, [2, 1]).Apply(new double[2], 1, new double[2]));
        Assert.Throws<ArgumentException>(() => new SimplexConstraint().Apply(Array.Empty<double>(), 1, Array.Empty<double>()));
    }

    // ---------- 補助 ----------

    private static double Phi(IProximalOperator h, double[] x, double[] v, double t)
    {
        double sum = 0;
        for (int i = 0; i < x.Length; i++)
        {
            sum += (x[i] - v[i]) * (x[i] - v[i]);
        }

        return h.Evaluate(x) + sum / (2 * t);
    }

    private static double[] RandomPoint(Random random, int n, double scale) =>
        Enumerable.Range(0, n).Select(_ => scale * (2 * random.NextDouble() - 1)).ToArray();

    internal static Matrix<double> RandomMatrix(int rows, int columns, int seed)
    {
        var random = new Random(seed);
        var a = new Matrix<double>(rows, columns);
        for (int j = 0; j < columns; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                a[i, j] = 2 * random.NextDouble() - 1;
            }
        }

        return a;
    }

    // 並べ替えによる単体への射影のしきい値（Held, Wolfe &amp; Crowder 1974 の形）
    private static double SortingThreshold(double[] v, double total)
    {
        var u = v.OrderByDescending(a => a).ToArray();
        double sum = 0, tau = 0;
        for (int j = 0; j < u.Length; j++)
        {
            sum += u[j];
            double candidate = (sum - total) / (j + 1);
            if (u[j] - candidate > 0)
            {
                tau = candidate;
            }
        }

        return tau;
    }
}
