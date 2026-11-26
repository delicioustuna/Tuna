using Tuna.Integration;

namespace Tuna.Tests.Integration;

/// <summary>1変数の適応型ガウス・クロンロッド求積（<see cref="Quadrature.Integrate(Func{double, double}, double, double, IntegrationOptions?)"/>）</summary>
public class QuadratureTests
{
    private const double Epsilon = 2.220446049250313e-16;
    private const double Inf = double.PositiveInfinity;

    /// <summary>名前（参照値の名前）・被積分関数・端点・区切り点</summary>
    public sealed record Case(string Name, Func<double, double> F, double A, double B, double[] Breakpoints)
    {
        public override string ToString() => Name;
    }

    private static readonly Case[] Cases =
    [
        new("Exp", Math.Exp, 0, 1, []),
        new("Cos100", x => Math.Cos(100 * x), 0, 1, []),
        new("Runge", x => 1 / (1 + 25 * x * x), -1, 1, []),
        new("Peak", x => 1 / ((x - 0.3) * (x - 0.3) + 1e-4), 0, 1, []),
        new("InverseSqrt", x => 1 / Math.Sqrt(x), 0, 1, []),
        new("Power09", x => Math.Pow(x, -0.9), 0, 1, []),
        new("Log", Math.Log, 0, 1, []),
        new("XLogX", x => x * Math.Log(x), 0, 1, []),
        new("SqrtKink", x => Math.Sqrt(Math.Abs(x - 0.3)), 0, 1, [0.3]),
        new("AbsKink", x => Math.Abs(x - 1.0 / 3), 0, 1, [1.0 / 3]),
        new("Step", Math.Floor, 0, 3, [2, 1]),
        new("LogAbs", x => Math.Log(Math.Abs(x)), -1, 1, [0]),
        new("Sinc", x => Math.Sin(x) / x, 0, 20 * Math.PI, []),
        new("Gaussian", x => Math.Exp(-x * x), -Inf, Inf, []),
        new("Cauchy", x => 1 / (1 + x * x), -Inf, Inf, []),
        new("ShiftedGaussian", x => Math.Exp(-(x - 50) * (x - 50)), -Inf, Inf, [50]),
        new("FarLorentzian", x => 1 / (1 + (x - 1000) * (x - 1000)), -Inf, Inf, [1000]),
        new("GammaHalf", x => Math.Exp(-x) / Math.Sqrt(x), 0, Inf, []),
        new("GammaFour", x => x * x * x * Math.Exp(-x), 0, Inf, []),
        new("GammaTenth", x => Math.Pow(x, -0.9) * Math.Exp(-x), 0, Inf, []),
        new("Tail15", x => Math.Pow(x, -1.5), 1, Inf, []),
        new("Tail12", x => Math.Pow(1 + x, -1.2), 0, Inf, []),
        new("DampedCos", x => Math.Exp(-x) * Math.Cos(x), 0, Inf, []),
        new("LowerExp", Math.Exp, -Inf, 0, []),
        new("LowerInverseSquare", x => 1 / (x * x), -Inf, -1, []),
        new("NormalTail", x => Math.Exp(-0.5 * x * x) / Math.Sqrt(2 * Math.PI), 3, Inf, []),
    ];

    public static TheoryData<Case, double> CasesWithTolerances()
    {
        var data = new TheoryData<Case, double>();
        foreach (var c in Cases)
        {
            data.Add(c, 1e-8);
            // ∫ cos(100x) は ∫|f| の 1% ほどまで打ち消し合い、相対 1e-12 は丸め誤差（50ε·∫|f|）より小さい
            if (c.Name != "Cos100")
            {
                data.Add(c, 1e-12);
            }
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(CasesWithTolerances))]
    public void ConvergesToTheReferenceWithinTheErrorEstimate(Case c, double tolerance)
    {
        double expected = IntegrationReferences.OneDimensional[c.Name];
        var r = Quadrature.Integrate(c.F, c.A, c.B, c.Breakpoints, new IntegrationOptions { RelativeTolerance = tolerance });

        Assert.Equal(IntegrationTermination.Tolerance, r.Termination);
        Assert.True(r.Converged);
        Assert.True(r.ErrorEstimate <= tolerance * Math.Abs(r.Value), $"estimate {r.ErrorEstimate:E3}");
        // 誤差の見積もりは実際の誤差を上から押さえる（値そのものの丸めの分だけ余裕を持たせる）
        double error = Math.Abs(r.Value - expected);
        Assert.True(error <= r.ErrorEstimate + 4 * Epsilon * Math.Abs(expected), $"error {error:E3}, estimate {r.ErrorEstimate:E3}");
        Assert.Equal(0, r.Evaluations % 15);
    }

    [Fact]
    public void ReversedLimitsNegate()
    {
        var forward = Quadrature.Integrate(Math.Exp, 0, 1);
        var backward = Quadrature.Integrate(Math.Exp, 1, 0);
        Assert.Equal(-forward.Value, backward.Value);
        Assert.Equal(forward.ErrorEstimate, backward.ErrorEstimate);

        var tail = Quadrature.Integrate(x => Math.Exp(-x), Inf, 0);
        Assert.Equal(-1, tail.Value, 1e-12);
        var line = Quadrature.Integrate(x => Math.Exp(-x * x), Inf, -Inf);
        Assert.Equal(-Math.Sqrt(Math.PI), line.Value, 1e-12);
    }

    [Theory]
    [InlineData(0.0, 0.0)]
    [InlineData(2.5, 2.5)]
    [InlineData(double.PositiveInfinity, double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity, double.NegativeInfinity)]
    public void EmptyIntervalIsZeroWithoutEvaluations(double a, double b)
    {
        var r = Quadrature.Integrate(_ => throw new InvalidOperationException(), a, b);
        Assert.Equal(new IntegrationResult(0, 0, 0, IntegrationTermination.Tolerance), r);
    }

    [Fact]
    public void PolynomialsUpToDegree23AreExactOnOneInterval()
    {
        // 1回の則（15点）だけで止める。クロンロッド則は次数23まで正確
        var once = new IntegrationOptions { MaxEvaluations = 15 };
        for (int k = 0; k <= 23; k++)
        {
            int power = k;
            var r = Quadrature.Integrate(x => Math.Pow(x, power), -1, 1, once);
            double expected = power % 2 == 0 ? 2.0 / (power + 1) : 0;
            Assert.Equal(15, r.Evaluations);
            Assert.True(Math.Abs(r.Value - expected) <= 8 * Epsilon, $"degree {power}: {r.Value:R}");
        }

        var degree24 = Quadrature.Integrate(x => Math.Pow(x, 24), -1, 1, once);
        Assert.True(Math.Abs(degree24.Value - 2.0 / 25) > 1e-12);
    }

    [Fact]
    public void BreakpointsAreNotEvaluatedAndDuplicatesAreIgnored()
    {
        var seen = new List<double>();
        double F(double x)
        {
            seen.Add(x);
            return Math.Abs(x - 0.5);
        }

        var r = Quadrature.Integrate(F, 0, 1, [0.5, 0.5, 0, 1]);
        Assert.Equal(0.25, r.Value, 1e-15);
        Assert.Equal(30, r.Evaluations);
        Assert.DoesNotContain(0.5, seen);
        Assert.DoesNotContain(0.0, seen);
        Assert.DoesNotContain(1.0, seen);
    }

    [Fact]
    public void EndpointSingularitiesAreResolvedNearZero()
    {
        // [0, ∞) は [0, 1] を変換せずに積分するので、0 の特異性は 1e-300 の近くまで分けて追える
        var zero = Quadrature.Integrate(x => Math.Exp(-x) / Math.Sqrt(x), 0, Inf, new IntegrationOptions { RelativeTolerance = 1e-12 });
        Assert.True(zero.Converged);
        Assert.Equal(Math.Sqrt(Math.PI), zero.Value, 1e-12);

        // 0 から離れた端点（7）の近くは倍精度で 1e-15 程度までしか分けられない。収束しないことを報告し、値は誤差の見積もりの範囲に入る
        var seven = Quadrature.Integrate(x => Math.Exp(-(x - 7)) / Math.Sqrt(x - 7), 7, Inf);
        Assert.Equal(IntegrationTermination.IntervalTooSmall, seven.Termination);
        Assert.True(Math.Abs(seven.Value - Math.Sqrt(Math.PI)) <= seven.ErrorEstimate);
        Assert.True(seven.ErrorEstimate < 1e-5);
    }

    [Fact]
    public void FarEndpointUsesTheTransformOnly()
    {
        // 端点が大きくて a + 1 = a になる場合は、変換だけを使う。質量は t ≈ 1e-17 にあり、t = 0 の側の細かい倍精度で追える
        var far = Quadrature.Integrate(x => Math.Exp(-(x - 1e17) * 1e-17), 1e17, Inf);
        Assert.True(far.Converged);
        Assert.Equal(1e17, far.Value, 1e17 * 1e-10);
    }

    [Fact]
    public void CancellationStopsAtTheRoundoffLevel()
    {
        // ∫ sin = 0 の相対誤差は保証できない。値は丸め誤差（50ε·∫|f|）の範囲で正しい
        var r = Quadrature.Integrate(Math.Sin, -Math.PI, Math.PI);
        Assert.Equal(IntegrationTermination.RoundoffLimited, r.Termination);
        Assert.False(r.Converged);
        Assert.True(Math.Abs(r.Value) <= r.ErrorEstimate);
        Assert.True(r.ErrorEstimate < 1e-12);

        // 絶対許容誤差を与えれば収束とみなす
        var absolute = Quadrature.Integrate(Math.Sin, -Math.PI, Math.PI, new IntegrationOptions { AbsoluteTolerance = 1e-12 });
        Assert.True(absolute.Converged);

        // 相対許容誤差が 50ε より小さいと、正の被積分関数でも丸め誤差で止まる
        var tight = Quadrature.Integrate(Math.Exp, 0, 1, new IntegrationOptions { RelativeTolerance = 1e-15 });
        Assert.Equal(IntegrationTermination.RoundoffLimited, tight.Termination);
        Assert.Equal(Math.E - 1, tight.Value, 1e-15);
    }

    [Fact]
    public void DivergentIntegralStopsWhenTheIntervalCannotBeSplit()
    {
        var r = Quadrature.Integrate(x => 1 / x, 0, 1);
        Assert.Equal(IntegrationTermination.IntervalTooSmall, r.Termination);
        Assert.False(r.Converged);
        Assert.True(r.ErrorEstimate > 1);
    }

    [Fact]
    public void SingularityAwayFromZeroIsReportedWhenItCannotBeResolved()
    {
        // 0.5 の近くは倍精度で 1e-16 程度までしか分けられず、|x − 0.5|^(−1/2) の質量 2√δ が残る。値は誤差の見積もりの範囲で正しい
        var r = Quadrature.Integrate(x => 1 / Math.Sqrt(Math.Abs(x - 0.5)), 0, 1, [0.5]);
        Assert.Equal(IntegrationTermination.IntervalTooSmall, r.Termination);
        Assert.True(Math.Abs(r.Value - 2 * Math.Sqrt(2)) <= r.ErrorEstimate);
        Assert.True(r.ErrorEstimate < 1e-5);
    }

    [Fact]
    public void NonFiniteValueStops()
    {
        // 区切り点として渡さない特異点に中心の節点が当たる
        var r = Quadrature.Integrate(x => 1 / Math.Sqrt(Math.Abs(x - 0.5)), 0, 1);
        Assert.Equal(IntegrationTermination.NonFiniteValue, r.Termination);
        Assert.True(double.IsNaN(r.Value));
        Assert.True(double.IsNaN(r.ErrorEstimate));
        Assert.Equal(15, r.Evaluations);

        var nan = Quadrature.Integrate(x => x > 0.9 ? double.NaN : x, 0, 1);
        Assert.Equal(IntegrationTermination.NonFiniteValue, nan.Termination);

        // 発散する裾では変換後の値が溢れる
        var divergent = Quadrature.Integrate(_ => 1.0, 0, Inf);
        Assert.Equal(IntegrationTermination.NonFiniteValue, divergent.Termination);
    }

    [Fact]
    public void EvaluationLimitStopsWithTheBestEstimate()
    {
        var r = Quadrature.Integrate(x => Math.Sin(1 / x), 0, 1, new IntegrationOptions { MaxEvaluations = 1000 });
        Assert.Equal(IntegrationTermination.MaxEvaluations, r.Termination);
        Assert.True(r.Evaluations <= 1000);
        Assert.True(r.Evaluations > 1000 - 30);
        Assert.Equal(0.5040670619069283, r.Value, r.ErrorEstimate);
    }

    [Fact]
    public void ZeroIntegrandConverges()
    {
        var r = Quadrature.Integrate(_ => 0.0, -3, 5);
        Assert.Equal(new IntegrationResult(0, 0, 15, IntegrationTermination.Tolerance), r);
    }

    [Fact]
    public void StructIntegrandGivesTheSameBitsAsTheDelegate()
    {
        var viaStruct = Quadrature.Integrate(new Shifted(0.25), 0, Inf);
        var viaDelegate = Quadrature.Integrate(x => Math.Exp(-x) / Math.Sqrt(Math.Abs(x - 0.25) + 1e-3), 0, Inf);
        Assert.Equal(viaDelegate, viaStruct);
        Assert.Equal(viaDelegate, Quadrature.Integrate(x => Math.Exp(-x) / Math.Sqrt(Math.Abs(x - 0.25) + 1e-3), 0, Inf));
    }

    private readonly struct Shifted(double center) : IIntegrand
    {
        public double Evaluate(double x) => Math.Exp(-x) / Math.Sqrt(Math.Abs(x - center) + 1e-3);
    }

    [Fact]
    public void InvalidArgumentsThrow()
    {
        Assert.Throws<ArgumentNullException>(() => Quadrature.Integrate((Func<double, double>)null!, 0, 1));
        Assert.Throws<ArgumentException>(() => Quadrature.Integrate(Math.Exp, double.NaN, 1));
        Assert.Throws<ArgumentException>(() => Quadrature.Integrate(Math.Exp, 0, double.NaN));
        Assert.Throws<ArgumentException>(() => Quadrature.Integrate(Math.Exp, 0, 1, [double.NaN]));
        Assert.Throws<ArgumentException>(() => Quadrature.Integrate(Math.Exp, 0, Inf, [Inf]));
        Assert.Throws<ArgumentOutOfRangeException>(() => Quadrature.Integrate(Math.Exp, 0, 1, [1.5]));
        Assert.Throws<ArgumentOutOfRangeException>(() => Quadrature.Integrate(Math.Exp, 0, 1, new IntegrationOptions { RelativeTolerance = -1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => Quadrature.Integrate(Math.Exp, 0, 1, new IntegrationOptions { RelativeTolerance = double.NaN }));
        Assert.Throws<ArgumentOutOfRangeException>(() => Quadrature.Integrate(Math.Exp, 0, 1, new IntegrationOptions { AbsoluteTolerance = Inf }));
        Assert.Throws<ArgumentOutOfRangeException>(() => Quadrature.Integrate(Math.Exp, 0, 1, new IntegrationOptions { MaxEvaluations = 14 }));
        // (−∞, ∞) は4区間（裾2つと [−1, 0]・[0, 1]）から始めるので 60 回が要る
        Assert.Throws<ArgumentOutOfRangeException>(() => Quadrature.Integrate(Math.Exp, -Inf, Inf, new IntegrationOptions { MaxEvaluations = 59 }));
        Assert.Equal(60, Quadrature.Integrate(x => Math.Exp(-x * x), -Inf, Inf, new IntegrationOptions { MaxEvaluations = 60 }).Evaluations);
    }
}
