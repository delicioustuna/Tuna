using Tuna.Integration;
using Tuna.LinearAlgebra;

namespace Tuna.Tests.Integration;

/// <summary>多次元の適応型求積（2・3次元はガウス・クロンロッドの直積、4次元以上は Genz–Malik）</summary>
public class CubatureTests
{
    private const double Epsilon = 2.220446049250313e-16;
    private const double Inf = double.PositiveInfinity;

    public static TheoryData<GenzCase, double> SmoothCases()
    {
        var data = new TheoryData<GenzCase, double>();
        foreach (var c in GenzFunction.Cases("Oscillatory", "ProductPeak", "CornerPeak", "Gaussian").Where(c => c.Dimension <= 5))
        {
            // 4次元以上の誤差の見積もりは次数5の則の誤差（控えめ）なので、許容誤差を緩める
            data.Add(c, c.Dimension switch { <= 3 => 1e-10, 4 => 1e-5, _ => 1e-4 });
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(SmoothCases))]
    public void SmoothGenzFunctionsConvergeWithinTheErrorEstimate(GenzCase c, double tolerance)
    {
        var lower = new double[c.Dimension];
        var upper = Enumerable.Repeat(1.0, c.Dimension).ToArray();
        var r = Cubature.Integrate(new GenzFunction(c), lower, upper, new IntegrationOptions { RelativeTolerance = tolerance, MaxEvaluations = 4_000_000 });

        Assert.Equal(IntegrationTermination.Tolerance, r.Termination);
        double error = Math.Abs(r.Value - c.Value);
        Assert.True(error <= r.ErrorEstimate + 16 * Epsilon * Math.Abs(c.Value), $"error {error:E3}, estimate {r.ErrorEstimate:E3}");
        Assert.True(error <= tolerance * Math.Abs(c.Value) * 1.0001, $"error {error:E3}");
    }

    [Theory]
    [MemberData(nameof(NonSmoothCases))]
    public void NonSmoothGenzFunctionsAreReportedHonestly(GenzCase c)
    {
        // 折れ目・段差は領域を細かく分けるので、上限までに収束しないことがある。値は誤差の見積もりの範囲に入る
        var lower = new double[c.Dimension];
        var upper = Enumerable.Repeat(1.0, c.Dimension).ToArray();
        var r = Cubature.Integrate(new GenzFunction(c), lower, upper, new IntegrationOptions { RelativeTolerance = 1e-6, MaxEvaluations = 200_000 });

        Assert.True(r.Termination is IntegrationTermination.Tolerance or IntegrationTermination.MaxEvaluations, r.Termination.ToString());
        Assert.True(Math.Abs(r.Value - c.Value) <= r.ErrorEstimate, $"error {Math.Abs(r.Value - c.Value):E3}, estimate {r.ErrorEstimate:E3}");
        Assert.True(Math.Abs(r.Value - c.Value) <= 1e-3 * Math.Abs(c.Value));
    }

    public static TheoryData<GenzCase> NonSmoothCases() => new(GenzFunction.Cases("Continuous", "Discontinuous").Where(c => c.Dimension <= 3));

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void ProductRuleIsExactForDegree23InEachAxis(int d)
    {
        // 1回の則（15^d 点）だけで止める
        long points = (long)Math.Pow(15, d);
        var once = new IntegrationOptions { MaxEvaluations = points };
        int[] powers = [22, 20, 14];
        var r = Cubature.Integrate(x =>
        {
            double p = 1;
            for (int i = 0; i < x.Length; i++)
            {
                p *= Math.Pow(x[i], powers[i]);
            }

            return p;
        }, new double[d], Enumerable.Repeat(1.0, d).ToArray(), once);

        double expected = 1;
        for (int i = 0; i < d; i++)
        {
            expected /= powers[i] + 1;
        }

        Assert.Equal(points, r.Evaluations);
        Assert.True(Math.Abs(r.Value - expected) <= 16 * Epsilon * expected, $"{r.Value:R} vs {expected:R}");
    }

    [Theory]
    [InlineData(4, 57)]
    [InlineData(5, 93)]
    [InlineData(6, 149)]
    public void GenzMalikRuleIsExactForDegree7(int d, long points)
    {
        var once = new IntegrationOptions { MaxEvaluations = points };
        var lower = new double[d];
        var upper = Enumerable.Range(0, d).Select(i => 1.0 + 0.5 * i).ToArray();

        // 次数7の単項式 x₀²·x₁³·x₂²（残りの軸は 1）と、次数6の x₀⁶
        var r7 = Cubature.Integrate(x => x[0] * x[0] * x[1] * x[1] * x[1] * x[2] * x[2], lower, upper, once);
        double expected7 = Math.Pow(upper[0], 3) / 3 * Math.Pow(upper[1], 4) / 4 * Math.Pow(upper[2], 3) / 3;
        for (int i = 3; i < d; i++)
        {
            expected7 *= upper[i];
        }

        Assert.Equal(points, r7.Evaluations);
        Assert.True(Math.Abs(r7.Value - expected7) <= 64 * Epsilon * expected7, $"{r7.Value:R} vs {expected7:R}");

        var r6 = Cubature.Integrate(x => Math.Pow(x[0], 6), lower, upper, once);
        double expected6 = Math.Pow(upper[0], 7) / 7;
        for (int i = 1; i < d; i++)
        {
            expected6 *= upper[i];
        }

        Assert.True(Math.Abs(r6.Value - expected6) <= 64 * Epsilon * expected6);

        // 次数8 は正確でない
        var r8 = Cubature.Integrate(x => Math.Pow(x[0], 8), lower, upper, once);
        double expected8 = Math.Pow(upper[0], 9) / 9;
        for (int i = 1; i < d; i++)
        {
            expected8 *= upper[i];
        }

        Assert.True(Math.Abs(r8.Value - expected8) > 1e-6 * expected8);
    }

    [Fact]
    public void InfiniteLimitsAreTransformedPerAxis()
    {
        var plane = Cubature.Integrate(x => Math.Exp(-x[0] * x[0] - x[1] * x[1]), [-Inf, -Inf], [Inf, Inf]);
        Assert.True(plane.Converged);
        Assert.True(Math.Abs(plane.Value - Math.PI) <= plane.ErrorEstimate);

        var quadrant = Cubature.Integrate(x => Math.Exp(-x[0] - 2 * x[1]), [0.0, 0.0], [Inf, Inf]);
        Assert.True(quadrant.Converged);
        Assert.Equal(0.5, quadrant.Value, 1e-12);

        var halfStrip = Cubature.Integrate(x => Math.Exp(x[0]) * Math.Exp(-x[1] * x[1]), [-Inf, 1.0], [0.0, Inf]);
        Assert.True(halfStrip.Converged);
        Assert.Equal(Math.Sqrt(Math.PI) / 2 * XMath.Special.Erfc(1), halfStrip.Value, 1e-12);

        // 4次元（Genz–Malik）: 標準正規の密度の積の全空間の積分 = 1
        var space = Cubature.Integrate(x =>
        {
            double s = 0;
            for (int i = 0; i < 4; i++)
            {
                s += x[i] * x[i];
            }

            return Math.Exp(-0.5 * s) / (4 * Math.PI * Math.PI);
        }, [-Inf, -Inf, -Inf, -Inf], [Inf, Inf, Inf, Inf], new IntegrationOptions { RelativeTolerance = 1e-4 });
        Assert.True(space.Converged);
        Assert.Equal(1, space.Value, 1e-4);
        Assert.True(Math.Abs(space.Value - 1) <= space.ErrorEstimate);

        // 3次元（直積の則）。(−∞, ∞) の軸は [−1, 1] と両側の裾の3つから始める（3³ = 27 領域）
        var cube = Cubature.Integrate(x => Math.Exp(-0.5 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2])) / Math.Pow(2 * Math.PI, 1.5),
            [-Inf, -Inf, -Inf], [Inf, Inf, Inf], new IntegrationOptions { RelativeTolerance = 1e-4 });
        Assert.True(cube.Converged);
        Assert.True(Math.Abs(cube.Value - 1) <= cube.ErrorEstimate);
    }

    [Fact]
    public void TooSmallBudgetIsRejectedBeforeAnyEvaluation()
    {
        // 最初の領域の数（(−∞, ∞) の軸 k 本で 3^k）× 1回の則の点の数に届かない上限は、領域を作る前に例外にする
        int evaluations = 0;
        double Count(ReadOnlyVectorSpan<double> _)
        {
            evaluations++;
            return 1;
        }

        Assert.Throws<ArgumentOutOfRangeException>(() => Cubature.Integrate(Count, [-Inf, -Inf, -Inf], [Inf, Inf, Inf],
            new IntegrationOptions { MaxEvaluations = 27 * 3375 - 1 }));
        // 4次元（Genz–Malik の則、57 点）で 3⁴ = 81 領域
        Assert.Throws<ArgumentOutOfRangeException>(() => Cubature.Integrate(Count, [-Inf, -Inf, -Inf, -Inf], [Inf, Inf, Inf, Inf],
            new IntegrationOptions { MaxEvaluations = 81 * 57 - 1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => Cubature.Integrate(Count, [-Inf, -Inf, -Inf, -Inf], [Inf, Inf, Inf, Inf],
            new IntegrationOptions { MaxEvaluations = 0 }));
        Assert.Equal(0, evaluations);

        var r = Cubature.Integrate(Count, [-Inf, 0.0, 0.0, 0.0], [Inf, 1.0, 1.0, 1.0], new IntegrationOptions { MaxEvaluations = 3 * 57 });
        Assert.Equal(3 * 57, r.Evaluations);
        Assert.Equal(3 * 57, evaluations);
    }

    [Fact]
    public void BoxesWhoseVolumeProductLeavesTheRangeAreRejected()
    {
        // 則は幅を前の軸から順に掛ける。1e-200·1e-200 は 0 に丸まり、1e200·1e200 は溢れるので、評価する前に例外にする
        int evaluations = 0;
        double One(ReadOnlyVectorSpan<double> _)
        {
            evaluations++;
            return 1;
        }

        Assert.Throws<ArgumentOutOfRangeException>(() => Cubature.Integrate(One, [0.0, 0, 0, 0], [1e-200, 1e-200, 1e200, 1e200]));
        Assert.Throws<ArgumentOutOfRangeException>(() => Cubature.Integrate(One, [0.0, 0], [1e200, 1e200]));
        Assert.Throws<ArgumentOutOfRangeException>(() => Cubature.Integrate(One, [-1e308, 0], [1e308, 1]));
        Assert.Equal(0, evaluations);

        // 途中の積が範囲に収まる順なら積分できる
        var r = Cubature.Integrate(One, [0.0, 0, 0, 0], [1e-200, 1e200, 1e-200, 1e200]);
        Assert.True(r.Converged);
        Assert.Equal(1, r.Value, 1e-14);
    }

    [Fact]
    public void ReversedAxesFlipTheSignAndEmptyAxesGiveZero()
    {
        var forward = Cubature.Integrate(x => x[0] * Math.Exp(x[1]), [0.0, 0.0], [1.0, 1.0]);
        var oneReversed = Cubature.Integrate(x => x[0] * Math.Exp(x[1]), [1.0, 0.0], [0.0, 1.0]);
        var bothReversed = Cubature.Integrate(x => x[0] * Math.Exp(x[1]), [1.0, 1.0], [0.0, 0.0]);
        Assert.Equal(-forward.Value, oneReversed.Value);
        Assert.Equal(forward.Value, bothReversed.Value);
        Assert.Equal((Math.E - 1) / 2, forward.Value, 1e-14);

        var empty = Cubature.Integrate(_ => throw new InvalidOperationException(), [0.0, 2.0, 0.0], [1.0, 2.0, Inf]);
        Assert.Equal(new IntegrationResult(0, 0, 0, IntegrationTermination.Tolerance), empty);
    }

    [Fact]
    public void TerminationReasons()
    {
        // 打ち消し合う積分は丸め誤差で止まる
        var cancel = Cubature.Integrate(x => Math.Sin(x[0] + x[1]), [-Math.PI, -Math.PI], [Math.PI, Math.PI]);
        Assert.Equal(IntegrationTermination.RoundoffLimited, cancel.Termination);
        Assert.True(Math.Abs(cancel.Value) <= cancel.ErrorEstimate);

        // 評価の回数の上限。1回の分割（2回の則）ができなければ止まる
        var limited = Cubature.Integrate(x => x[0] + x[1] < 1 ? 1.0 : 0.0, [0.0, 0.0], [1.0, 1.0], new IntegrationOptions { MaxEvaluations = 5000 });
        Assert.Equal(IntegrationTermination.MaxEvaluations, limited.Termination);
        Assert.True(limited.Evaluations <= 5000);
        Assert.Equal(0, limited.Evaluations % 225);

        var nan = Cubature.Integrate(x => x[0] > 0.9 ? double.NaN : 1.0, [0.0, 0.0], [1.0, 1.0]);
        Assert.Equal(IntegrationTermination.NonFiniteValue, nan.Termination);
        Assert.True(double.IsNaN(nan.Value));

        // 原点に積分できない特異点: 領域を分けきれなくなる
        var divergent = Cubature.Integrate(x => 1 / (x[0] * x[0] + x[1] * x[1]), [0.0, 0.0], [1.0, 1.0]);
        Assert.False(divergent.Converged);
        Assert.True(divergent.Termination is IntegrationTermination.IntervalTooSmall or IntegrationTermination.MaxEvaluations or IntegrationTermination.NonFiniteValue);
    }

    [Fact]
    public void ResultsAreReproducibleAndTheStructPathMatchesTheDelegate()
    {
        var c = IntegrationReferences.Genz.First(g => g.Family == "Gaussian" && g.Dimension == 4);
        var lower = new double[4];
        var upper = new[] { 1.0, 1.0, 1.0, 1.0 };
        var options = new IntegrationOptions { RelativeTolerance = 1e-7 };
        var f = new GenzFunction(c);
        var first = Cubature.Integrate(f, lower, upper, options);
        var second = Cubature.Integrate(f, lower, upper, options);
        var viaDelegate = Cubature.Integrate((ReadOnlyVectorSpan<double> x) => f.Evaluate(x), lower, upper, options);
        Assert.Equal(first, second);
        Assert.Equal(first, viaDelegate);
    }

    [Fact]
    public void InvalidArgumentsThrow()
    {
        Assert.Throws<ArgumentNullException>(() => Cubature.Integrate((Func<ReadOnlyVectorSpan<double>, double>)null!, [0.0], [1.0]));
        Assert.Throws<ArgumentException>(() => Cubature.Integrate(_ => 1.0, [0.0, 0.0], [1.0]));
        Assert.Throws<ArgumentOutOfRangeException>(() => Cubature.Integrate(_ => 1.0, [], []));
        Assert.Throws<ArgumentOutOfRangeException>(() => Cubature.Integrate(_ => 1.0, new double[21], new double[21]));
        Assert.Throws<ArgumentException>(() => Cubature.Integrate(_ => 1.0, [double.NaN], [1.0]));
        Assert.Throws<ArgumentException>(() => Cubature.Integrate(_ => 1.0, [0.0], [double.NaN]));
        Assert.Throws<ArgumentOutOfRangeException>(() => Cubature.Integrate(_ => 1.0, [0.0], [1.0], new IntegrationOptions { AbsoluteTolerance = -1 }));
        // 10次元の Genz–Malik の則は 2^10 + 2·100 + 20 + 1 = 1,245 点
        var ten = Enumerable.Repeat(1.0, 10).ToArray();
        Assert.Throws<ArgumentOutOfRangeException>(() => Cubature.Integrate(_ => 1.0, new double[10], ten, new IntegrationOptions { MaxEvaluations = 1244 }));
        var r = Cubature.Integrate(_ => 1.0, new double[10], ten, new IntegrationOptions { MaxEvaluations = 1245 });
        Assert.Equal(1245, r.Evaluations);
        Assert.Equal(1, r.Value, 1e-14);
    }
}
