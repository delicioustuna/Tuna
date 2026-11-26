using Tuna.LinearAlgebra;
using Tuna.Optimization;
using static Tuna.Tests.Optimization.GradientTestFunctions;

namespace Tuna.Tests.Optimization;

[Trait("Optimization", "Gradient")]
public class GradientHelperTests
{
    // ---------- 差分 ----------

    [Fact]
    public void CentralDifferencesMatchTheAnalyticGradient()
    {
        var f = new Rosenbrock();
        var numeric = new FiniteDifferenceGradient(f, 5);
        double[] x = [-1.2, 1.0, 0.3, 2.5, -40.0];
        var expected = new double[5];
        var actual = new double[5];
        double value = numeric.Evaluate(x, actual);
        Assert.Equal(f.Evaluate(x, expected), value);
        for (int i = 0; i < 5; i++)
        {
            Assert.True(Math.Abs(actual[i] - expected[i]) <= 1e-8 * Math.Max(1, Math.Abs(expected[i])), $"component {i}");
        }

        Assert.Equal(11, numeric.FunctionEvaluations);   // 2n + 1
        Assert.Equal(value, numeric.Evaluate(x));
        Assert.Equal(12, numeric.FunctionEvaluations);
    }

    [Fact]
    public void DelegateAndInterfaceGiveTheSameDifferences()
    {
        var f = new Rosenbrock();
        double[] x = [0.5, -0.5, 1.5];
        var a = new double[3];
        var b = new double[3];
        new FiniteDifferenceGradient(f, 3).Evaluate(x, a);
        new FiniteDifferenceGradient(p => f.Evaluate(p), 3).Evaluate(x, b);
        Assert.Equal(a, b);
    }

    [Fact]
    public void DifferencesStayInsideTheBounds()
    {
        // 範囲の外では NaN を返す関数（√x と √(1 − y) を含む）でも、境界の上と近くで勾配が求まる
        double Function(ReadOnlyVectorSpan<double> p)
        {
            if (p[0] < 0 || p[1] > 1 || p[2] < 2 || p[2] > 2 + 1e-9)
            {
                return double.NaN;
            }

            return Math.Sqrt(p[0] + 1) + Math.Sqrt(2 - p[1]) + p[2] * p[2];
        }

        var numeric = new FiniteDifferenceGradient(Function, 4, [0.0, double.NegativeInfinity, 2.0, 3.0], [double.PositiveInfinity, 1.0, 2 + 1e-9, 3.0]);
        var gradient = new double[4];
        foreach (double[] x in new[] { new[] { 0.0, 1.0, 2.0, 3.0 }, new[] { 1e-9, 1 - 1e-9, 2 + 1e-9, 3.0 } })
        {
            numeric.Evaluate(x, gradient);
            Assert.Equal(0.5 / Math.Sqrt(x[0] + 1), gradient[0], 1e-7);    // 片側差分
            Assert.Equal(-0.5 / Math.Sqrt(2 - x[1]), gradient[1], 1e-7);
            Assert.Equal(4.0, gradient[2], 1e-5);                            // 幅 1e-9 の割線
            Assert.Equal(0.0, gradient[3]);                                  // 固定された変数
        }
    }

    [Fact]
    public void FiniteDifferenceRejectsInvalidArguments()
    {
        Assert.Throws<ArgumentNullException>(() => new FiniteDifferenceGradient((Func<ReadOnlyVectorSpan<double>, double>)null!, 2));
        Assert.Throws<ArgumentNullException>(() => new FiniteDifferenceGradient((IObjectiveFunction)null!, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => new FiniteDifferenceGradient(_ => 0, 0));
        Assert.Throws<ArgumentException>(() => new FiniteDifferenceGradient(_ => 0, 2, [0.0], []));
        Assert.Throws<ArgumentException>(() => new FiniteDifferenceGradient(_ => 0, 1, [1.0], [0.0]));
        var numeric = new FiniteDifferenceGradient(_ => 0, 2);
        Assert.Throws<ArgumentException>(() => numeric.Evaluate(new double[3]));
        Assert.Throws<ArgumentException>(() => numeric.Evaluate(new double[2], new double[1]));
    }

    // ---------- 勾配の検査 ----------

    [Fact]
    public void GradientCheckSeparatesCorrectAndWrongGradients()
    {
        double[] x = [-1.2, 1.0, 0.8];
        Assert.True(GradientCheck.MaximumError(new Rosenbrock(), x) < 1e-8);

        // 1成分だけ符号を誤った勾配
        var f = new Rosenbrock();
        double wrong = GradientCheck.MaximumError(
            (p, g) =>
            {
                double value = f.Evaluate(p, g);
                g[1] = -g[1];
                return value;
            },
            x);
        Assert.True(wrong > 1e-2, $"error {wrong}");
        Assert.Throws<ArgumentException>(() => GradientCheck.MaximumError(new Rosenbrock(), []));
    }
}
