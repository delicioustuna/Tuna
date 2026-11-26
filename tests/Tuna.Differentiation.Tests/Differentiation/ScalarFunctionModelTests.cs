using Tuna.Differentiation;
using Tuna.LinearAlgebra;
using static Tuna.Tests.Differentiation.DifferentiationAssert;

namespace Tuna.Tests.Differentiation;

/// <summary>初等関数と特殊関数を組み合わせたモデルの勾配と再利用</summary>
public class ScalarFunctionModelTests
{
    [Theory]
    [InlineData(0.25, -0.4)]
    [InlineData(-0.5, 0.2)]
    [InlineData(1.0, -1.0)]
    public void ComposedGradientAgreesWithRichardsonDifferences(double x, double y)
    {
        var tape = new Tape();
        double[] point = [x, y];
        double[] gradient = new double[2];
        double value = Evaluate(tape, point, gradient);
        EqualBits(Model(x, y).Value, value);
        Close(Richardson(a => Model(a, y).Value, x), gradient[0],
            relativeTolerance: 2e-8, absoluteTolerance: 2e-8, context: "x");
        Close(Richardson(b => Model(x, b).Value, y), gradient[1],
            relativeTolerance: 2e-8, absoluteTolerance: 2e-8, context: "y");
    }

    [Fact]
    public void ReusingComposedScalarFunctionsAllocatesNoManagedBytes()
    {
        var tape = new Tape();
        double[] point = [0.25, -0.4];
        double[] gradient = new double[2];
        for (int i = 0; i < 100; i++)
        {
            Evaluate(tape, point, gradient);
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 100; i++)
        {
            Evaluate(tape, point, gradient);
        }

        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
        Assert.All(gradient, derivative => Assert.True(double.IsFinite(derivative)));
    }

    private static double Evaluate(Tape tape, double[] point, double[] gradient)
    {
        tape.Reset();
        ReadOnlySpan<ScalarNode> inputs = tape.Leaves(new ReadOnlyVectorSpan<double>(point));
        return tape.Gradient(Model(inputs[0], inputs[1]), new VectorSpan<double>(gradient));
    }

    private static ScalarNode Model(ScalarNode x, ScalarNode y)
    {
        ScalarNode a = 2 + XMath.Exp(x);
        ScalarNode b = 3 + XMath.Exp(y);
        ScalarNode z = x - y;
        ReadOnlySpan<ScalarNode> terms = [x, y, z];
        return XMath.Pow(a, b) + XMath.LogAddExp(x, y) + XMath.LogSubExp(a, y) + XMath.LogSumExp(terms)
            + XMath.Special.LogGamma(a) + XMath.Special.Digamma(b) + XMath.Special.LogBeta(a, b)
            + XMath.Special.Erf(z) + XMath.Special.Erfc(z)
            + XMath.Special.NormalPdf(z) + XMath.Special.NormalLogPdf(z)
            + XMath.Special.NormalCdf(z) + XMath.Special.NormalSf(z)
            + XMath.Special.NormalLogCdf(z) + XMath.Special.NormalLogSf(z);
    }
}
