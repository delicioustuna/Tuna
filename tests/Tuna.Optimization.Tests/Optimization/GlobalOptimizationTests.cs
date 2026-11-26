using System.Numerics;
using Tuna.Optimization;
using Tuna.Optimization.Global;

namespace Tuna.Tests.Optimization;

[Trait("Optimization", "Global")]
public class GlobalOptimizationTests
{
    // ---------- 区間演算 ----------

    [Fact]
    public void IntervalArithmeticEnclosesTheExactResultWithinOneUlp()
    {
        var random = new Random(41);
        for (int trial = 0; trial < 2000; trial++)
        {
            double a = RandomDouble(random), b = RandomDouble(random);
            Interval x = a, y = b;

            Interval sum = x + y;
            Assert.True(Compare(Exact(sum.Lower), Add(Exact(a), Exact(b))) <= 0 && Compare(Add(Exact(a), Exact(b)), Exact(sum.Upper)) <= 0);
            AssertWithinOneUlp(sum);

            Interval difference = x - y;
            var exactDifference = Add(Exact(a), Exact(-b));
            Assert.True(Compare(Exact(difference.Lower), exactDifference) <= 0 && Compare(exactDifference, Exact(difference.Upper)) <= 0);
            AssertWithinOneUlp(difference);

            Interval product = x * y;
            var exactProduct = Multiply(Exact(a), Exact(b));
            Assert.True(Compare(Exact(product.Lower), exactProduct) <= 0 && Compare(exactProduct, Exact(product.Upper)) <= 0);
            AssertWithinOneUlp(product);

            // b·lower ≤ a ≤ b·upper（b > 0）か、その逆（b < 0）
            Interval quotient = x / y;
            int sign = Math.Sign(b);
            Assert.True(sign * Compare(Multiply(Exact(quotient.Lower), Exact(b)), Exact(a)) <= 0);
            Assert.True(sign * Compare(Exact(a), Multiply(Exact(quotient.Upper), Exact(b))) <= 0);
            AssertWithinOneUlp(quotient);

            double c = Math.Abs(a);
            Interval root = Interval.Sqrt(c);
            Assert.True(Compare(Multiply(Exact(root.Lower), Exact(root.Lower)), Exact(c)) <= 0);
            Assert.True(Compare(Exact(c), Multiply(Exact(root.Upper), Exact(root.Upper))) <= 0);
            AssertWithinOneUlp(root);
        }
    }

    [Fact]
    public void IntervalArithmeticDoesNotWidenExactResults()
    {
        Assert.Equal(new Interval(3), (Interval)1 + 2);
        Assert.Equal(new Interval(1.5), (Interval)3 * 0.5);
        Assert.Equal(new Interval(0.25), (Interval)1 / 4);
        Assert.Equal(new Interval(2), Interval.Sqrt(4));
        Assert.Equal(new Interval(1), Interval.Exp(0));
        Assert.Equal(new Interval(0), Interval.Log(1));
        Assert.Equal(new Interval(-5, 7), new Interval(-1, 2) * new Interval(2, 3) - new Interval(2, 3) + new Interval(1, 3));
        Assert.Equal(new Interval(-1, 1), new Interval(1, 2) - new Interval(1, 2));
    }

    // 区間の中点は非正規化数でも区間内に収まり、巨大な端点でもオーバーフローしない。
    [Theory]
    [InlineData(1, 1)]
    [InlineData(3, 3)]
    [InlineData(-1, -1)]
    [InlineData(1, 2)]
    [InlineData(-3, 5)]
    public void IntervalMidpointOfSubnormalEndpointsStaysInside(int lower, int upper)
    {
        var interval = new Interval(lower * double.Epsilon, upper * double.Epsilon);
        Assert.True(interval.Contains(interval.Midpoint), $"{interval}, midpoint={interval.Midpoint:R}");
        if (lower == upper)
        {
            Assert.Equal(lower * double.Epsilon, interval.Midpoint);
        }
    }

    [Fact]
    public void IntervalMidpointOfHugeEndpointsDoesNotOverflow()
    {
        Assert.Equal(double.MaxValue, new Interval(double.MaxValue).Midpoint);
        Assert.Equal(0.0, new Interval(-double.MaxValue, double.MaxValue).Midpoint);
        Assert.Equal(0.75 * double.MaxValue, new Interval(0.5 * double.MaxValue, double.MaxValue).Midpoint);
        Assert.Equal(double.NegativeInfinity, new Interval(double.NegativeInfinity, 1).Midpoint);
        Assert.Equal(0.0, Interval.Entire.Midpoint);
        Assert.True(double.IsNaN(Interval.Empty.Midpoint));
    }

    [Fact]
    public void IntervalElementaryFunctionsHandleDomainsAndSigns()
    {
        Assert.Equal(new Interval(0, 2), Interval.Sqrt(new Interval(-1, 4)));
        Assert.True(Interval.Sqrt(new Interval(-2, -1)).IsEmpty);
        Assert.Equal(new Interval(double.NegativeInfinity, 0), Interval.Log(new Interval(0, 1)));
        Assert.True(Interval.Log(new Interval(-3, 0)).IsEmpty);
        Assert.Equal(new Interval(0.5, double.PositiveInfinity), Interval.Reciprocal(new Interval(0, 2)));
        Assert.Equal(new Interval(double.NegativeInfinity, -0.5), Interval.Reciprocal(new Interval(-2, 0)));
        Assert.Equal(Interval.Entire, Interval.Reciprocal(new Interval(-1, 2)));
        Assert.True(Interval.Reciprocal(0).IsEmpty);
        Assert.True((new Interval(1, 2) / 0).IsEmpty);
        Assert.Equal(new Interval(1, double.PositiveInfinity), new Interval(1, 2) / new Interval(0, 1));
        Assert.Equal(Interval.Entire, new Interval(1, 2) / new Interval(-1, 1));
        Assert.Equal(new Interval(0, 9), Interval.Square(new Interval(-2, 3)));
        Assert.Equal(new Interval(0, 9), Interval.Pow(new Interval(-2, 3), 2));
        Assert.Equal(new Interval(0, 3), Interval.Abs(new Interval(-2, 3)));
        Assert.Equal(new Interval(2, 3), Interval.Abs(new Interval(-3, -2)));
        Assert.Equal(new Interval(-1, 5), Interval.Max(new Interval(-3, 5), new Interval(-1, 2)));
        Assert.Equal(new Interval(-3, 2), Interval.Min(new Interval(-3, 5), new Interval(-1, 2)));
        Assert.Equal(new Interval(1), Interval.Pow(new Interval(-3, 5), 0));

        Interval cube = Interval.Pow(new Interval(-2, 3), 3);
        Assert.True(cube.Contains(-8) && cube.Contains(27) && cube.Lower > -8.0001 && cube.Upper < 27.0001);
        Interval inverseSquare = Interval.Pow(new Interval(2, 4), -2);
        Assert.True(inverseSquare.Contains(1.0 / 16) && inverseSquare.Contains(0.25) && inverseSquare.Lower > 0.0624 && inverseSquare.Upper < 0.2501);
        Interval exp = Interval.Exp(new Interval(-1, 1));
        Assert.True(exp.Lower < Math.Exp(-1) && Math.Exp(1) < exp.Upper && exp.Lower > 0);
        Assert.Equal(new Interval(0, 1), Interval.Exp(new Interval(double.NegativeInfinity, 0)));

        Assert.True((Interval.Empty + 1).IsEmpty);
        Assert.True(Interval.Exp(Interval.Empty).IsEmpty);
        Assert.True(((Interval)double.NaN).IsEmpty);
        Assert.True(((Interval)double.PositiveInfinity).IsEmpty);
        Assert.Equal(new Interval(0, 0), new Interval(0) * new Interval(1, double.PositiveInfinity));
        Assert.Equal("[-1, 2.5]", new Interval(-1, 2.5).ToString());
        Assert.Equal(new Interval(1, 2), new Interval(0, 2).Intersect(new Interval(1, 3)));
        Assert.True(new Interval(0, 1).Intersect(new Interval(2, 3)).IsEmpty);
        Assert.Equal(new Interval(0, 3), new Interval(0, 1).Hull(new Interval(2, 3)));
    }

    [Fact]
    public void IntervalConstructorRejectsInvalidBounds()
    {
        Assert.Throws<ArgumentException>(() => new Interval(2, 1));
        Assert.Throws<ArgumentException>(() => new Interval(double.NaN, 1));
        Assert.Throws<ArgumentException>(() => new Interval(double.PositiveInfinity, double.PositiveInfinity));
        Assert.Throws<ArgumentException>(() => new Interval(double.NegativeInfinity, double.NegativeInfinity));
        Assert.Throws<ArgumentException>(() => new Interval(double.PositiveInfinity));
        Assert.Throws<ArgumentOutOfRangeException>(() => Interval.Pow(new Interval(1, 2), int.MinValue));
    }

    [Fact]
    public void IntervalEvaluationContainsFloatingPointValues()
    {
        // 浮動小数点の評価の途中の値は、各段の区間に入る（四則と平方根は正しく丸められ、初等関数は 1 ulp 以内）
        var random = new Random(42);
        var f = new Mixed();
        double[] lo = [0.5, -1.5, 0.2], hi = [2, 1, 3];
        for (int trial = 0; trial < 300; trial++)
        {
            var (boxLower, boxUpper) = RandomSubBox(random, lo, hi);
            var intervals = new Interval[3];
            for (int i = 0; i < 3; i++)
            {
                intervals[i] = new Interval(boxLower[i], boxUpper[i]);
            }

            Interval range = f.Evaluate<Interval>(intervals);
            for (int s = 0; s < 20; s++)
            {
                double value = ValueAt(f, RandomPoint(random, boxLower, boxUpper));
                Assert.True(range.Contains(value), $"{value} is outside {range}");
            }
        }
    }

    // ---------- McCormick 緩和 ----------

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(10)]
    [InlineData(11)]
    [InlineData(12)]
    [InlineData(13)]
    [InlineData(14)]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(17)]
    [InlineData(18)]
    [InlineData(19)]
    [InlineData(20)]
    [InlineData(21)]
    public void RelaxationsSandwichTheFunctionWithValidSubgradients(int which)
    {
        AssertSandwich(new Elementary(which), [-1.7, 0.4], [2.3, 3.1], seed: 100 + which);
    }

    [Fact]
    public void RelaxationsOfACompositeFunctionAreValid()
    {
        AssertSandwich(new Mixed(), [0.5, -1.5, 0.2], [2, 1, 3], seed: 43);
        AssertSandwich(new SixHumpCamel(), [-3, -2], [3, 2], seed: 44);
        AssertSandwich(new GoldsteinPrice(), [-2, -2], [2, 2], seed: 45);
    }

    [Fact]
    public void BilinearRelaxationIsTheMcCormickEnvelope()
    {
        // xy の凸包絡 max(yL·x + xL·y − xL·yL, yU·x + xU·y − xU·yU) と凹包絡 min(yU·x + xL·y − xL·yU, yL·x + xU·y − xU·yL)
        double xL = -1, xU = 2, yL = 0.5, yU = 3, px = 0.3, py = 1.1;
        var x = McCormick.Variable(new Interval(xL, xU), px, 0, 2);
        var y = McCormick.Variable(new Interval(yL, yU), py, 1, 2);
        McCormick w = x * y;
        double convex = Math.Max(yL * px + xL * py - xL * yL, yU * px + xU * py - xU * yU);
        double concave = Math.Min(yU * px + xL * py - xL * yU, yL * px + xU * py - xU * yL);
        Assert.Equal(convex, w.Convex, 1e-15);
        Assert.Equal(concave, w.Concave, 1e-15);
        Assert.Equal(new Interval(-3, 6), w.Range);
        Assert.Equal(2, w.ConvexSubgradient.Length);
    }

    [Fact]
    public void RelaxationsTightenQuadraticallyAsTheBoxShrinks()
    {
        // exp の割線と関数の差は箱の幅の2乗で縮む
        double previous = double.NaN;
        for (int k = 0; k < 6; k++)
        {
            double w = Math.Pow(0.5, k);
            McCormick e = McCormick.Exp(McCormick.Variable(new Interval(1 - w, 1 + w), 1, 0, 1));
            double gap = e.Concave - e.Convex;
            Assert.True(gap > 0);
            if (k > 0)
            {
                Assert.InRange(previous / gap, 3.5, 4.5);
            }

            previous = gap;
        }

        // 0 を中心とする箱の x² は中点で凸な緩和が正確
        McCormick square = McCormick.Square(McCormick.Variable(new Interval(-0.25, 0.25), 0, 0, 1));
        Assert.Equal(0, square.Convex);
        Assert.Equal(0.0625, square.Concave, 1e-16);
    }

    [Fact]
    public void ConstantsAndUndefinedValuesPropagate()
    {
        McCormick c = 2.5;
        Assert.Equal(new Interval(2.5), c.Range);
        Assert.Equal(2.5, c.Convex);
        Assert.True(c.ConvexSubgradient.IsEmpty);

        var x = McCormick.Variable(new Interval(-2, -1), -1.5, 0, 1);
        Assert.True(McCormick.Sqrt(x).Range.IsEmpty);
        Assert.True(double.IsNaN((McCormick.Log(x) + 1).Convex));
        Assert.True(((McCormick)double.NaN).Range.IsEmpty);

        // 0 を跨ぐ 1/x は区間の端（ここでは ±∞）の定数
        var z = McCormick.Variable(new Interval(-1, 2), 0.5, 0, 1);
        McCormick r = McCormick.Reciprocal(z);
        Assert.Equal(Interval.Entire, r.Range);
        Assert.Equal(double.NegativeInfinity, r.Convex);
        Assert.True(r.ConvexSubgradient.IsEmpty);

        Assert.Throws<ArgumentException>(() => McCormick.Variable(new Interval(0, 1), 2, 0, 1));
        Assert.Throws<ArgumentException>(() => McCormick.Variable(Interval.Empty, 0, 0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => McCormick.Variable(new Interval(0, 1), 0.5, 1, 1));
        Assert.Throws<ArgumentException>(() => McCormick.Variable(new Interval(0, 1), 0.5, 0, 1) + McCormick.Variable(new Interval(0, 1), 0.5, 0, 2));
    }

    // ---------- 値と勾配 ----------

    [Fact]
    public void GradientDualGivesExactGradients()
    {
        var objective = new FactorableObjective<Mixed>(new Mixed(), 3);
        var random = new Random(46);
        for (int trial = 0; trial < 20; trial++)
        {
            double[] x = RandomPoint(random, [0.5, -1.5, 0.2], [2, 1, 3]);
            Assert.True(GradientCheck.MaximumError(objective, x) < 1e-6);
            Assert.Equal(ValueAt(new Mixed(), x), objective.Evaluate(x), 1e-14);
        }

        // 1変数の初等関数の導関数
        var v = GradientDual.Variable(0.7, 0, 1);
        Assert.Equal(Math.Exp(0.7), GradientDual.Exp(v).Gradient[0], 1e-15);
        Assert.Equal(1 / 0.7, GradientDual.Log(v).Gradient[0], 1e-15);
        Assert.Equal(0.5 / Math.Sqrt(0.7), GradientDual.Sqrt(v).Gradient[0], 1e-15);
        Assert.Equal(-1 / 0.49, GradientDual.Reciprocal(v).Gradient[0], 1e-14);
        Assert.Equal(-3 * Math.Pow(0.7, -4), GradientDual.Pow(v, -3).Gradient[0], 1e-12);
        Assert.Equal(1, GradientDual.Abs(-v).Gradient[0]);
        Assert.True(((GradientDual)3).Gradient.IsEmpty);
    }

    // ---------- 分枝限定法 ----------

    [Fact]
    public void BranchAndBoundFindsTheSixHumpCamelMinimum()
    {
        var bb = new BranchAndBound([-3, -2], [3, 2]);
        var result = bb.Minimize(new SixHumpCamel());
        Assert.Equal(BranchAndBoundTermination.GapTolerance, result.Termination);
        Assert.True(result.Converged);
        Assert.Equal(-1.0316284534898774, result.Value, 1e-9);
        Assert.InRange(result.Gap, 0, 1e-6 * 1.0316284534898774 + 1e-12);
        Assert.Equal(0.0898420131003, Math.Abs(result.Point[0]), 1e-5);
        Assert.Equal(0.7126564030207, Math.Abs(result.Point[1]), 1e-5);
        Assert.True(result.Point[0] * result.Point[1] < 0);
        Assert.True(result.LocalSearches >= 1);
    }

    [Fact]
    public void BranchAndBoundFindsTheGoldsteinPriceMinimum()
    {
        var bb = new BranchAndBound([-2, -2], [2, 2]);
        var result = bb.Minimize(new GoldsteinPrice());
        Assert.True(result.Converged, $"{result.Termination} after {result.Nodes} nodes, gap {result.Gap}");
        Assert.Equal(3, result.Value, 1e-9);
        Assert.Equal(0, result.Point[0], 1e-5);
        Assert.Equal(-1, result.Point[1], 1e-5);
        Assert.True(result.LowerBound <= result.Value);
    }

    [Fact]
    public void BranchAndBoundFindsTheStyblinskiTangMinimum()
    {
        const int n = 3;
        // 1変数の最小点は 4x³ − 32x + 5 = 0 の負の根
        double root = -2.9;
        for (int k = 0; k < 50; k++)
        {
            root -= (4 * root * root * root - 32 * root + 5) / (12 * root * root - 32);
        }

        double expected = n * 0.5 * (Math.Pow(root, 4) - 16 * root * root + 5 * root);
        var bb = new BranchAndBound([-5, -5, -5], [5, 5, 5]);
        var result = bb.Minimize(new StyblinskiTang());
        Assert.True(result.Converged, $"{result.Termination} after {result.Nodes} nodes, gap {result.Gap}");
        Assert.Equal(expected, result.Value, 1e-8);
        for (int i = 0; i < n; i++)
        {
            Assert.Equal(root, result.Point[i], 1e-5);
        }
    }

    [Fact]
    public void BranchAndBoundFindsTheRosenbrockMinimum()
    {
        // 2乗の和なので区間の下界が 0 になり、局所的な最小化の点で証明が済む
        var bb = new BranchAndBound([-2, -1], [2, 3]);
        var result = bb.Minimize(new Rosenbrock());
        Assert.True(result.Converged, $"{result.Termination} after {result.Nodes} nodes, gap {result.Gap}");
        Assert.InRange(result.Value, 0, 1e-9);
        Assert.InRange(result.LowerBound, -1e-6, result.Value);
        Assert.Equal(1, result.Point[0], 1e-4);
        Assert.Equal(1, result.Point[1], 1e-4);
    }

    [Fact]
    public void RelaxationsReduceTheNumberOfNodes()
    {
        // 区間だけの下界は同じ変数の相関を無視するので、最小点の近くで箱が減らない
        var withRelaxation = new BranchAndBound([-3, -2], [3, 2]).Minimize(new SixHumpCamel());
        var intervalOnly = new BranchAndBound([-3, -2], [3, 2], new BranchAndBoundOptions { UseRelaxation = false, MaxNodes = 20000 })
            .Minimize(new SixHumpCamel());
        Assert.True(withRelaxation.Converged);
        Assert.Equal(withRelaxation.Value, intervalOnly.Value, 1e-9);
        Assert.True(intervalOnly.LowerBound <= intervalOnly.Value);
        Assert.True(!intervalOnly.Converged || withRelaxation.Nodes < intervalOnly.Nodes, $"{withRelaxation.Nodes} vs {intervalOnly.Nodes}");
        Assert.True(withRelaxation.Gap < intervalOnly.Gap);
    }

    [Fact]
    public void BranchAndBoundWithoutLocalSearchStillProvesTheBound()
    {
        var result = new BranchAndBound([-3, -2], [3, 2], new BranchAndBoundOptions { LocalSearch = false, AbsoluteTolerance = 1e-4 })
            .Minimize(new SixHumpCamel());
        Assert.True(result.Converged);
        Assert.Equal(0, result.LocalSearches);
        Assert.InRange(result.Value, -1.0316284534898774, -1.0316284534898774 + 1e-4 + 1e-9);
        Assert.InRange(result.Gap, 0, 1e-4 + 1e-12);
    }

    [Fact]
    public void BranchAndBoundHandlesNonSmoothFunctions()
    {
        // |x − 0.3| + max(y, −y/2) + min(xy, 0.2)
        var bb = new BranchAndBound([-1, -1], [1, 1], new BranchAndBoundOptions { AbsoluteTolerance = 1e-7 });
        var result = bb.Minimize(new NonSmooth());
        Assert.True(result.Converged, $"{result.Termination} after {result.Nodes} nodes, gap {result.Gap}");

        // 格子の上の最小値以下で、下界とも矛盾しない
        double gridMinimum = double.PositiveInfinity;
        for (int i = 0; i <= 400; i++)
        {
            for (int j = 0; j <= 400; j++)
            {
                gridMinimum = Math.Min(gridMinimum, ValueAt(new NonSmooth(), [-1 + i / 200.0, -1 + j / 200.0]));
            }
        }

        // 局所的な最小化は折れ目で止まりうるが、上界と下界は許容誤差の範囲で格子の最小を挟む
        Assert.True(result.Value <= gridMinimum + 1e-7);
        Assert.True(result.LowerBound <= gridMinimum && result.Value - result.LowerBound <= 1e-7 + 1e-12);
    }

    [Fact]
    public void BranchAndBoundMinimizesOverTheDomainOfDefinition()
    {
        // −√(x − 1) は [1, 4] でだけ定義され、最小は x = 4 で −√3
        var result = new BranchAndBound([0], [4]).Minimize(new ShiftedRoot());
        Assert.True(result.Converged);
        Assert.Equal(-Math.Sqrt(3), result.Value, 1e-9);
        Assert.Equal(4, result.Point[0], 1e-6);

        var infeasible = new BranchAndBound([0], [0.5]).Minimize(new ShiftedRoot());
        Assert.Equal(BranchAndBoundTermination.NoFeasiblePoint, infeasible.Termination);
        Assert.True(double.IsNaN(infeasible.Value));
        Assert.False(infeasible.Converged);
    }

    [Fact]
    public void BranchAndBoundStopsAtTheNodeLimit()
    {
        var result = new BranchAndBound([-3, -2], [3, 2], new BranchAndBoundOptions { MaxNodes = 3 }).Minimize(new SixHumpCamel());
        Assert.Equal(BranchAndBoundTermination.MaxNodes, result.Termination);
        Assert.Equal(3, result.Nodes);
        Assert.True(result.LowerBound <= result.Value);
        Assert.False(result.Converged);
    }

    // 非正規化数の幅を持つ箱でも実行可能解を返すことを検証する。
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BranchAndBoundReturnsAFeasibleSubnormalFixedPoint(bool relaxation)
    {
        var result = new BranchAndBound([double.Epsilon], [double.Epsilon],
            new BranchAndBoundOptions { UseRelaxation = relaxation, LocalSearch = false }).Minimize(new Identity());
        Assert.Equal(double.Epsilon, result.Point[0]);
        Assert.Equal(double.Epsilon, result.Value);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BranchAndBoundSplitsSubnormalBoxesInside(bool relaxation)
    {
        // 幅が数個の非正規化数しかない箱でも、中点と参照点が箱の外に出ない
        var result = new BranchAndBound([-3 * double.Epsilon], [5 * double.Epsilon],
            new BranchAndBoundOptions { UseRelaxation = relaxation, LocalSearch = false }).Minimize(new Identity());
        Assert.InRange(result.Point[0], -3 * double.Epsilon, 5 * double.Epsilon);
        Assert.InRange(result.Value, -3 * double.Epsilon, 5 * double.Epsilon);
    }

    [Fact]
    public void BranchAndBoundValidatesItsArguments()
    {
        Assert.Throws<ArgumentException>(() => new BranchAndBound([], []));
        Assert.Throws<ArgumentException>(() => new BranchAndBound([0, 0], [1]));
        Assert.Throws<ArgumentException>(() => new BranchAndBound([0], [double.PositiveInfinity]));
        Assert.Throws<ArgumentException>(() => new BranchAndBound([1], [0]));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BranchAndBound([0], [1], new BranchAndBoundOptions { AbsoluteTolerance = -1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BranchAndBound([0], [1], new BranchAndBoundOptions { RelativeTolerance = double.NaN }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BranchAndBound([0], [1], new BranchAndBoundOptions { MaxNodes = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BranchAndBound([0], [1], new BranchAndBoundOptions { MinimumRelativeWidth = 1 }));

        using var canceled = new CancellationTokenSource();
        canceled.Cancel();
        Assert.Throws<OperationCanceledException>(() => new BranchAndBound([-3, -2], [3, 2]).Minimize(new SixHumpCamel(), canceled.Token));
    }

    // ---------- 補助 ----------

    private static void AssertSandwich<TFunction>(TFunction f, double[] lower, double[] upper, int seed) where TFunction : IFactorableFunction
    {
        var random = new Random(seed);
        int n = lower.Length;
        for (int trial = 0; trial < 200; trial++)
        {
            var (boxLower, boxUpper) = RandomSubBox(random, lower, upper);
            double[] p = RandomPoint(random, boxLower, boxUpper);
            if (trial % 3 == 0)
            {
                // 基準点が箱の端にあると、緩和が区間の端と一致して mid の選択が分かれる
                for (int i = 0; i < n; i++)
                {
                    p[i] = random.Next(2) == 0 ? boxLower[i] : boxUpper[i];
                }
            }

            var variables = new McCormick[n];
            for (int i = 0; i < n; i++)
            {
                variables[i] = McCormick.Variable(new Interval(boxLower[i], boxUpper[i]), p[i], i, n);
            }

            McCormick m = f.Evaluate<McCormick>(variables);
            double fp = ValueAt(f, p);
            double scale = 1e-9 * (1 + Math.Abs(fp) + Math.Abs(m.Convex) + Math.Abs(m.Concave));
            Assert.False(m.Range.IsEmpty);
            Assert.True(m.Convex <= fp + scale && fp <= m.Concave + scale, $"trial {trial}: {m} at f(p) = {fp}");
            Assert.True(m.Range.Lower <= m.Convex && m.Concave <= m.Range.Upper);

            ReadOnlySpan<double> sv = m.ConvexSubgradient, sc = m.ConcaveSubgradient;
            for (int s = 0; s < 30; s++)
            {
                double[] x = RandomPoint(random, boxLower, boxUpper);
                if (s < 1 << n)
                {
                    // 頂点も確かめる
                    for (int i = 0; i < n; i++)
                    {
                        x[i] = ((s >> i) & 1) == 0 ? boxLower[i] : boxUpper[i];
                    }
                }

                double fx = ValueAt(f, x);
                double below = m.Convex, above = m.Concave;
                for (int i = 0; i < n; i++)
                {
                    below += (sv.IsEmpty ? 0 : sv[i]) * (x[i] - p[i]);
                    above += (sc.IsEmpty ? 0 : sc[i]) * (x[i] - p[i]);
                }

                double tolerance = 1e-9 * (1 + Math.Abs(fx) + Math.Abs(below) + Math.Abs(above));
                if (double.IsFinite(below))
                {
                    Assert.True(below <= fx + tolerance, $"trial {trial}: affine underestimate {below} > f(x) = {fx}");
                }

                if (double.IsFinite(above))
                {
                    Assert.True(fx <= above + tolerance, $"trial {trial}: affine overestimate {above} < f(x) = {fx}");
                }
            }
        }
    }

    private static double ValueAt<TFunction>(TFunction f, double[] x) where TFunction : IFactorableFunction
    {
        var values = new GradientDual[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            values[i] = x[i];
        }

        return f.Evaluate<GradientDual>(values).Value;
    }

    private static (double[] Lower, double[] Upper) RandomSubBox(Random random, double[] lower, double[] upper)
    {
        var lo = new double[lower.Length];
        var hi = new double[lower.Length];
        for (int i = 0; i < lower.Length; i++)
        {
            double a = lower[i] + (upper[i] - lower[i]) * random.NextDouble();
            double b = random.Next(4) == 0 ? a : lower[i] + (upper[i] - lower[i]) * random.NextDouble();
            lo[i] = Math.Min(a, b);
            hi[i] = Math.Max(a, b);
        }

        if (random.Next(5) == 0)
        {
            return ((double[])lower.Clone(), (double[])upper.Clone());
        }

        return (lo, hi);
    }

    private static double[] RandomPoint(Random random, double[] lower, double[] upper)
    {
        var x = new double[lower.Length];
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = Math.Clamp(lower[i] + (upper[i] - lower[i]) * random.NextDouble(), lower[i], upper[i]);
        }

        return x;
    }

    private static double RandomDouble(Random random)
    {
        double v = (random.NextDouble() * 2 - 1) * Math.Pow(2, random.Next(-40, 41));
        return v == 0 ? 1 : v;
    }

    private static void AssertWithinOneUlp(Interval x) =>
        Assert.True(x.Lower == x.Upper || Math.BitIncrement(x.Lower) == x.Upper, $"{x} is wider than one ulp");

    /// <summary>有限の double の正確な値 m·2^e</summary>
    private static (BigInteger Mantissa, int Exponent) Exact(double v)
    {
        long bits = BitConverter.DoubleToInt64Bits(v);
        int exponent = (int)((bits >> 52) & 0x7FF);
        long fraction = bits & 0xFFFFFFFFFFFFFL;
        if (exponent == 0)
        {
            exponent = 1;
        }
        else
        {
            fraction |= 1L << 52;
        }

        BigInteger mantissa = bits < 0 ? -(BigInteger)fraction : fraction;
        return (mantissa, exponent - 1075);
    }

    private static (BigInteger Mantissa, int Exponent) Add((BigInteger Mantissa, int Exponent) a, (BigInteger Mantissa, int Exponent) b)
    {
        int e = Math.Min(a.Exponent, b.Exponent);
        return ((a.Mantissa << (a.Exponent - e)) + (b.Mantissa << (b.Exponent - e)), e);
    }

    private static (BigInteger Mantissa, int Exponent) Multiply((BigInteger Mantissa, int Exponent) a, (BigInteger Mantissa, int Exponent) b) =>
        (a.Mantissa * b.Mantissa, a.Exponent + b.Exponent);

    private static int Compare((BigInteger Mantissa, int Exponent) a, (BigInteger Mantissa, int Exponent) b)
    {
        int e = Math.Min(a.Exponent, b.Exponent);
        return (a.Mantissa << (a.Exponent - e)).CompareTo(b.Mantissa << (b.Exponent - e));
    }

    // ---------- 関数 ----------

    /// <summary>x ∈ [−1.7, 2.3]、y ∈ [0.4, 3.1] の上の1つの演算</summary>
    private readonly struct Elementary(int which) : IFactorableFunction
    {
        public T Evaluate<T>(ReadOnlySpan<T> v) where T : IFactorableNumber<T>
        {
            T x = v[0], y = v[1];
            return which switch
            {
                0 => x * y,
                1 => x / y,
                2 => T.Exp(x),
                3 => T.Log(y),
                4 => T.Sqrt(y),
                5 => T.Square(x),
                6 => T.Pow(x, 3),
                7 => T.Pow(x, 5),
                8 => T.Pow(x, 4),
                9 => T.Pow(y, -2),
                10 => T.Reciprocal(y),
                11 => T.Reciprocal(x - 2.5),
                12 => T.Abs(x),
                13 => T.Max(x, y - 1),
                14 => T.Min(x, 1 - y),
                15 => T.Exp(x * y / 3),
                16 => T.Log(y) * x,
                17 => (x + 1.8) / y,
                18 => T.Sqrt(x + 1.7),
                19 => T.Pow(x - 0.3, 3) * y,
                20 => 2.5 / y - y / 3 + 1 - x * 0.5,
                21 => T.Max(T.Abs(x), T.Square(y) - 3) - T.Pow(x * 0.5 - y, 7),
                _ => throw new ArgumentOutOfRangeException(nameof(which)),
            };
        }
    }

    /// <summary>x ∈ [0.5, 2]、y ∈ [−1.5, 1]、z ∈ [0.2, 3] で定義される、すべての演算を含む関数</summary>
    private readonly struct Mixed : IFactorableFunction
    {
        public T Evaluate<T>(ReadOnlySpan<T> v) where T : IFactorableNumber<T>
        {
            T x = v[0], y = v[1], z = v[2];
            return x * y - T.Exp(0.5 * y) * T.Log(x + z) + T.Sqrt(z) / x + T.Pow(y, 3) - 2 * T.Pow(y - 0.2, 5)
                + T.Abs(y - 0.1) * z + T.Max(x, z) - T.Min(y, x * 0.3) + T.Reciprocal(x + y * 0.2) + T.Pow(z, -2) + T.Square(x - y);
        }
    }

    private readonly struct SixHumpCamel : IFactorableFunction
    {
        public T Evaluate<T>(ReadOnlySpan<T> v) where T : IFactorableNumber<T>
        {
            T x = v[0], y = v[1];
            T x2 = T.Square(x), y2 = T.Square(y);
            return (4 - 2.1 * x2 + T.Pow(x, 4) / 3) * x2 + x * y + (-4 + 4 * y2) * y2;
        }
    }

    private readonly struct GoldsteinPrice : IFactorableFunction
    {
        public T Evaluate<T>(ReadOnlySpan<T> v) where T : IFactorableNumber<T>
        {
            T x = v[0], y = v[1];
            T a = 1 + T.Square(x + y + 1) * (19 - 14 * x + 3 * T.Square(x) - 14 * y + 6 * x * y + 3 * T.Square(y));
            T b = 30 + T.Square(2 * x - 3 * y) * (18 - 32 * x + 12 * T.Square(x) + 48 * y - 36 * x * y + 27 * T.Square(y));
            return a * b;
        }
    }

    private readonly struct StyblinskiTang : IFactorableFunction
    {
        public T Evaluate<T>(ReadOnlySpan<T> v) where T : IFactorableNumber<T>
        {
            T sum = 0.0;
            foreach (T x in v)
            {
                sum += T.Pow(x, 4) - 16 * T.Square(x) + 5 * x;
            }

            return 0.5 * sum;
        }
    }

    private readonly struct Rosenbrock : IFactorableFunction
    {
        public T Evaluate<T>(ReadOnlySpan<T> v) where T : IFactorableNumber<T> =>
            100 * T.Square(v[1] - T.Square(v[0])) + T.Square(1 - v[0]);
    }

    private readonly struct NonSmooth : IFactorableFunction
    {
        public T Evaluate<T>(ReadOnlySpan<T> v) where T : IFactorableNumber<T> =>
            T.Abs(v[0] - 0.3) + T.Max(v[1], -0.5 * v[1]) + T.Min(v[0] * v[1], 0.2);
    }

    private readonly struct ShiftedRoot : IFactorableFunction
    {
        public T Evaluate<T>(ReadOnlySpan<T> v) where T : IFactorableNumber<T> => -T.Sqrt(v[0] - 1);
    }

    private readonly struct Identity : IFactorableFunction
    {
        public T Evaluate<T>(ReadOnlySpan<T> x) where T : IFactorableNumber<T> => x[0];
    }
}
