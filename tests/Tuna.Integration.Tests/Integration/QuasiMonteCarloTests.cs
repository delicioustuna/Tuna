using Tuna.Integration;
using Tuna.LinearAlgebra;
using Tuna.RandomSource;

namespace Tuna.Tests.Integration;

/// <summary>乱択化した Sobol' 列による準モンテカルロ積分</summary>
public class QuasiMonteCarloTests
{
    public static TheoryData<GenzCase> AllGenzCases() => new(IntegrationReferences.Genz);

    [Theory]
    [MemberData(nameof(AllGenzCases))]
    public void EstimatesCoverTheReference(GenzCase c)
    {
        // 8次元では値が被積分関数の大きさより桁違いに小さいもの（CornerPeak は 5e-7）があり、相対誤差の分散が大きいので許容誤差を緩める
        double tolerance = c.Dimension <= 5 ? 1e-5 : 1e-2;
        var r = QuasiMonteCarlo.Integrate(new GenzFunction(c), c.Dimension, new Philox4x32Random(seed: 20261028, streamId: (ulong)c.Dimension),
            new QuasiMonteCarloOptions { RelativeTolerance = tolerance });

        Assert.Equal(IntegrationTermination.Tolerance, r.Termination);
        Assert.True(r.StandardError <= tolerance * Math.Abs(r.Value));
        Assert.Equal(r.Replicates * r.PointsPerReplicate, r.Evaluations);
        // 推定値は不偏で、標準誤差は誤差の目安になる（複製 16 の標準誤差の推定のばらつきを見込んで 5 倍）
        double error = Math.Abs(r.Value - c.Value);
        Assert.True(error <= 5 * r.StandardError, $"error {error:E3}, standard error {r.StandardError:E3}");
    }

    [Fact]
    public void StandardErrorIsCalibratedAcrossIndependentRandomizations()
    {
        // 同じ問題を独立な乱数源 200 個で解き、標準化した誤差 z = (推定値 − 参照値)/標準誤差 の分布を見る
        var c = IntegrationReferences.Genz.First(g => g.Family == "Gaussian" && g.Dimension == 4);
        var options = new QuasiMonteCarloOptions { Replicates = 8, InitialPoints = 256, AbsoluteTolerance = 0, RelativeTolerance = 0, MaxEvaluations = 8 * 256 };
        int within2 = 0;
        double meanError = 0;
        const int runs = 200;
        for (int s = 0; s < runs; s++)
        {
            var r = QuasiMonteCarlo.Integrate(new GenzFunction(c), 4, new Philox4x32Random(seed: 7, streamId: (ulong)s), options);
            double z = (r.Value - c.Value) / r.StandardError;
            if (Math.Abs(z) <= 2)
            {
                within2++;
            }

            meanError += (r.Value - c.Value) / runs;
        }

        // t 分布（自由度 7）で |z| ≤ 2 は約 91%。偏りは 1 回の標準誤差より十分小さい
        Assert.InRange(within2, 160, 196);
        var single = QuasiMonteCarlo.Integrate(new GenzFunction(c), 4, new Philox4x32Random(seed: 7, streamId: 0), options);
        Assert.True(Math.Abs(meanError) < 0.3 * single.StandardError, $"bias {meanError:E3}, standard error {single.StandardError:E3}");
    }

    [Fact]
    public void ErrorDecreasesFasterThanMonteCarlo()
    {
        // 滑らかな被積分関数で点を 64 倍にすると、標準誤差はモンテカルロの 1/8 よりずっと小さくなる（およそ 1/64）
        var c = IntegrationReferences.Genz.First(g => g.Family == "ProductPeak" && g.Dimension == 3);
        QuasiMonteCarloResult Fixed(int points) => QuasiMonteCarlo.Integrate(new GenzFunction(c), 3, new Philox4x32Random(11),
            new QuasiMonteCarloOptions { InitialPoints = points, RelativeTolerance = 0, MaxEvaluations = 16L * points });
        var coarse = Fixed(1 << 10);
        var fine = Fixed(1 << 16);
        Assert.True(coarse.StandardError / fine.StandardError > 24, $"{coarse.StandardError:E3} → {fine.StandardError:E3}");
    }

    [Fact]
    public void FixedNumberOfPoints()
    {
        var options = new QuasiMonteCarloOptions { Replicates = 4, InitialPoints = 512, RelativeTolerance = 0, MaxEvaluations = 4 * 512 };
        var r = QuasiMonteCarlo.Integrate(x => x[0] * x[1], 2, new Philox4x32Random(3), options);
        Assert.Equal(IntegrationTermination.MaxEvaluations, r.Termination);
        Assert.Equal(512, r.PointsPerReplicate);
        Assert.Equal(2048, r.Evaluations);
        Assert.Equal(4, r.Replicates);
        Assert.Equal(0.25, r.Value, 1e-4);
    }

    [Fact]
    public void DoublingReusesThePreviousPoints()
    {
        // 1,024 点から2倍にした結果は、最初から 2,048 点で求めた結果とビット単位で一致する
        var c = IntegrationReferences.Genz.First(g => g.Family == "Oscillatory" && g.Dimension == 5);
        var doubled = QuasiMonteCarlo.Integrate(new GenzFunction(c), 5, new Philox4x32Random(5),
            new QuasiMonteCarloOptions { InitialPoints = 1024, RelativeTolerance = 0, MaxEvaluations = 16 * 2048 });
        var direct = QuasiMonteCarlo.Integrate(new GenzFunction(c), 5, new Philox4x32Random(5),
            new QuasiMonteCarloOptions { InitialPoints = 2048, RelativeTolerance = 0, MaxEvaluations = 16 * 2048 });
        Assert.Equal(2048, doubled.PointsPerReplicate);
        Assert.Equal(direct, doubled);
    }

    [Fact]
    public void ConsumesSixtyFourWordsPerDimensionAndReplicate()
    {
        var random = new Philox4x32Random(seed: 9, streamId: 2);
        QuasiMonteCarlo.Integrate(x => x[0], 3, random, new QuasiMonteCarloOptions { Replicates = 5 });
        var expected = new Philox4x32Random(seed: 9, streamId: 2);
        for (int i = 0; i < 5 * 3 * 64; i++)
        {
            expected.NextUInt64();
        }

        Assert.Equal(expected.NextUInt64(), random.NextUInt64());
    }

    [Fact]
    public void ResultsAreReproducibleAndTheStructPathMatchesTheDelegate()
    {
        var c = IntegrationReferences.Genz.First(g => g.Family == "CornerPeak" && g.Dimension == 8);
        var f = new GenzFunction(c);
        var a = QuasiMonteCarlo.Integrate(f, 8, new Philox4x32Random(1));
        var b = QuasiMonteCarlo.Integrate(f, 8, new Philox4x32Random(1));
        var viaDelegate = QuasiMonteCarlo.Integrate((ReadOnlyVectorSpan<double> x) => f.Evaluate(x), 8, new Philox4x32Random(1));
        Assert.Equal(a, b);
        Assert.Equal(a, viaDelegate);
        Assert.NotEqual(a.Value, QuasiMonteCarlo.Integrate(f, 8, new Philox4x32Random(2)).Value);
    }

    [Theory]
    [InlineData(0.5, 0.3, 0.4)]
    [InlineData(-0.6, 0.2, 0.1)]
    [InlineData(0.9, 0.85, 0.8)]
    public void TrivariateNormalOrthantProbability(double r12, double r13, double r23)
    {
        // Genz（1992）の変数変換: P(X > 0)（X ~ N(0, R)）を (0, 1)² の上の積分にする。閉形式は 1/8 + (asin r12 + asin r13 + asin r23)/(4π)
        double exact = 0.125 + (Math.Asin(r12) + Math.Asin(r13) + Math.Asin(r23)) / (4 * Math.PI);
        var integrand = new GenzOrthant(Correlation(3, (i, j) => (i, j) switch { (1, 0) => r12, (2, 0) => r13, _ => r23 }));
        var r = QuasiMonteCarlo.Integrate(integrand, 2, new Philox4x32Random(42), new QuasiMonteCarloOptions { AbsoluteTolerance = 1e-7, RelativeTolerance = 0 });
        Assert.True(r.Converged);
        Assert.True(Math.Abs(r.Value - exact) <= 5 * r.StandardError, $"{r.Value:R} vs {exact:R} (se {r.StandardError:E2})");
    }

    [Theory]
    [InlineData(5)]
    [InlineData(8)]
    public void EquicorrelatedNormalOrthantProbability(int n)
    {
        // 相関がすべて 1/2 の n 変量正規で P(X > 0) = 1/(n + 1)
        var integrand = new GenzOrthant(Correlation(n, (_, _) => 0.5));
        var r = QuasiMonteCarlo.Integrate(integrand, n - 1, new Philox4x32Random(seed: 3, streamId: (ulong)n),
            new QuasiMonteCarloOptions { AbsoluteTolerance = 1e-6, RelativeTolerance = 0 });
        Assert.True(r.Converged);
        Assert.True(Math.Abs(r.Value - 1.0 / (n + 1)) <= 5 * r.StandardError, $"{r.Value:R} (se {r.StandardError:E2})");
    }

    [Fact]
    public void NonFiniteValueStops()
    {
        var r = QuasiMonteCarlo.Integrate(x => 1 / (x[0] - x[0]), 2, new Philox4x32Random(1));
        Assert.Equal(IntegrationTermination.NonFiniteValue, r.Termination);
        Assert.True(double.IsNaN(r.Value));
        Assert.Equal(1, r.Evaluations);

        // 値は有限でも、複製の和（値 × 点の数）が溢れたら止める（上限で止まる設定でも同じ）
        foreach (long limit in new long[] { 1L << 24, 16 * 1024 })
        {
            var overflow = QuasiMonteCarlo.Integrate(_ => 1e306, 2, new Philox4x32Random(1), new QuasiMonteCarloOptions { MaxEvaluations = limit });
            Assert.Equal(IntegrationTermination.NonFiniteValue, overflow.Termination);
            Assert.True(double.IsNaN(overflow.Value));
            Assert.True(double.IsNaN(overflow.StandardError));
            Assert.Equal(16 * 1024, overflow.Evaluations);
        }
    }

    [Theory]
    [InlineData(1e-200)]
    [InlineData(1e200)]
    public void StandardErrorDoesNotDependOnTheScaleOfTheIntegrand(double scale)
    {
        // 偏差の2乗がアンダーフロー・オーバーフローする尺度でも、値と標準誤差は尺度 1 の結果の scale 倍で、同じ点の数で止まる
        var c = IntegrationReferences.Genz.First(g => g.Family == "Gaussian" && g.Dimension == 3);
        var unit = QuasiMonteCarlo.Integrate(new GenzFunction(c), 3, new Philox4x32Random(9));
        var scaled = QuasiMonteCarlo.Integrate(x => scale * new GenzFunction(c).Evaluate(x), 3, new Philox4x32Random(9));
        Assert.Equal(IntegrationTermination.Tolerance, unit.Termination);
        Assert.Equal(unit.Termination, scaled.Termination);
        Assert.Equal(unit.PointsPerReplicate, scaled.PointsPerReplicate);
        Assert.True(unit.StandardError > 0);
        Assert.Equal(1, scaled.Value / scale / unit.Value, 1e-14);
        Assert.Equal(1, scaled.StandardError / scale / unit.StandardError, 1e-12);
    }

    [Fact]
    public void InvalidArgumentsThrow()
    {
        var random = new Philox4x32Random(1);
        Assert.Throws<ArgumentNullException>(() => QuasiMonteCarlo.Integrate((Func<ReadOnlyVectorSpan<double>, double>)null!, 2, random));
        Assert.Throws<ArgumentNullException>(() => QuasiMonteCarlo.Integrate(_ => 1.0, 2, null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => QuasiMonteCarlo.Integrate(_ => 1.0, 0, random));
        Assert.Throws<ArgumentOutOfRangeException>(() => QuasiMonteCarlo.Integrate(_ => 1.0, 1112, random));
        Assert.Throws<ArgumentOutOfRangeException>(() => QuasiMonteCarlo.Integrate(_ => 1.0, 2, random, new QuasiMonteCarloOptions { Replicates = 1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => QuasiMonteCarlo.Integrate(_ => 1.0, 2, random, new QuasiMonteCarloOptions { InitialPoints = 1000 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => QuasiMonteCarlo.Integrate(_ => 1.0, 2, random, new QuasiMonteCarloOptions { InitialPoints = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => QuasiMonteCarlo.Integrate(_ => 1.0, 2, random, new QuasiMonteCarloOptions { RelativeTolerance = double.NaN }));
        Assert.Throws<ArgumentOutOfRangeException>(() => QuasiMonteCarlo.Integrate(_ => 1.0, 2, random, new QuasiMonteCarloOptions { AbsoluteTolerance = -1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => QuasiMonteCarlo.Integrate(_ => 1.0, 2, random,
            new QuasiMonteCarloOptions { Replicates = 4, InitialPoints = 1024, MaxEvaluations = 4095 }));
    }

    private static double[,] Correlation(int n, Func<int, int, double> offDiagonal)
    {
        var r = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            r[i, i] = 1;
            for (int j = 0; j < i; j++)
            {
                r[i, j] = r[j, i] = offDiagonal(i, j);
            }
        }

        return r;
    }

    /// <summary>
    /// Genz（1992）の変数変換で、正規分布の正の象限の確率を (0, 1)^(n−1) の上の積分にした被積分関数。
    /// Cholesky 因子 L で、wᵢ から yᵢ = Φ⁻¹(dᵢ + wᵢ(1 − dᵢ)) を順に作り、条件付きの確率 1 − dᵢ を掛ける
    /// </summary>
    private sealed class GenzOrthant : IMultivariateIntegrand
    {
        private readonly double[,] l;
        private readonly double[] y;

        public GenzOrthant(double[,] correlation)
        {
            int n = correlation.GetLength(0);
            l = new double[n, n];
            y = new double[n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    double s = correlation[i, j];
                    for (int k = 0; k < j; k++)
                    {
                        s -= l[i, k] * l[j, k];
                    }

                    l[i, j] = i == j ? Math.Sqrt(s) : s / l[j, j];
                }
            }
        }

        public double Evaluate(ReadOnlyVectorSpan<double> w)
        {
            int n = y.Length;
            double d = XMath.Special.NormalCdf(0);
            double product = 1 - d;
            for (int i = 1; i < n; i++)
            {
                y[i - 1] = XMath.Special.InverseNormalCdf(d + w[i - 1] * (1 - d));
                double s = 0;
                for (int j = 0; j < i; j++)
                {
                    s += l[i, j] * y[j];
                }

                d = XMath.Special.NormalCdf(-s / l[i, i]);
                product *= 1 - d;
            }

            return product;
        }
    }
}
