using Tuna;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;
using static Tuna.Tests.Differentiation.DifferentiationAssert;

namespace Tuna.Tests.Differentiation;

/// <summary>初等関数を組み合わせたモデルと記録領域の再利用</summary>
public class ElementaryModelTests
{
    private static readonly double[] Features = [-2, -0.5, 0, 0.75, 2];
    private static readonly double[] Observations = [2.5, 0.3, -0.4, -1, -0.5];

    [Theory]
    [InlineData(0.4, -0.7, 0.25)]
    [InlineData(-1.0, 0.25, -2.0)]
    [InlineData(0.0, 0.0, 2.0)]
    public void StudentTWithLogScaleAgreesWithHandwrittenGradient(double alpha, double beta, double logScale)
    {
        var tape = new Tape();
        double[] point = [alpha, beta, logScale];
        double[] gradient = new double[3];
        var p = tape.Leaves(new ReadOnlyVectorSpan<double>(point));
        double value = tape.Gradient(StudentT(p[0], p[1], p[2]), new VectorSpan<double>(gradient));
        EqualBits(value, StudentT(alpha, beta, logScale).Value);

        double[] expected = [-alpha / 100, -beta / 100, -logScale / 4];
        double scale = Math.Exp(logScale);
        for (int i = 0; i < Features.Length; i++)
        {
            double residual = (Observations[i] - alpha - beta * Features[i]) / scale;
            double derivative = 5 * residual / (4 + residual * residual) / scale;
            expected[0] += derivative;
            expected[1] += derivative * Features[i];
            expected[2] += -1 + 5 * residual * residual / (4 + residual * residual);
        }

        for (int i = 0; i < expected.Length; i++)
        {
            Close(expected[i], gradient[i], relativeTolerance: 5e-13, absoluteTolerance: 5e-13, context: $"gradient[{i}]");
        }
    }

    [Fact]
    public void ReusingEveryElementaryFunctionAllocatesNoManagedBytes()
    {
        var tape = new Tape();
        double[] point = [0.25, -0.4];
        double[] gradient = new double[2];
        for (int i = 0; i < 100; i++)
        {
            EvaluateEveryFunction(tape, point, gradient);
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 100; i++)
        {
            EvaluateEveryFunction(tape, point, gradient);
        }

        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(0, allocated);
        Assert.All(gradient, x => Assert.True(double.IsFinite(x)));
        EqualBits(EveryFunction(point[0], point[1]).Value, EvaluateEveryFunction(tape, point, gradient));
    }

    [Fact]
    public void SingularAndInvalidLocalDerivativesRespectTheZeroContributionRules()
    {
        var tape = new Tape();
        var x = tape.Leaf(0);
        var square = XMath.Square(x);
        tape.Backward(XMath.Sqrt(square));
        Assert.Equal(double.PositiveInfinity, tape.GetGradient(square));
        Assert.Equal(0, tape.GetGradient(x));

        tape.Reset();
        x = tape.Leaf(-1);
        var output = 3 * x;
        _ = XMath.Log(x);
        tape.Backward(output);
        Assert.Equal(3, tape.GetGradient(x));
    }

    private static ScalarNode StudentT(ScalarNode alpha, ScalarNode beta, ScalarNode logScale)
    {
        ScalarNode scale = XMath.Exp(logScale);
        ScalarNode value = -0.5 * alpha * alpha / 100 - 0.5 * beta * beta / 100 - 0.5 * logScale * logScale / 4;
        for (int i = 0; i < Features.Length; i++)
        {
            ScalarNode residual = (Observations[i] - alpha - beta * Features[i]) / scale;
            value += -logScale - 2.5 * XMath.Log1p(XMath.Square(residual) / 4);
        }

        return value;
    }

    private static double EvaluateEveryFunction(Tape tape, double[] point, double[] gradient)
    {
        tape.Reset();
        var p = tape.Leaves(new ReadOnlyVectorSpan<double>(point));
        return tape.Gradient(EveryFunction(p[0], p[1]), new VectorSpan<double>(gradient));
    }

    private static ScalarNode EveryFunction(ScalarNode x, ScalarNode y)
    {
        ScalarNode positive = 1 + XMath.Exp(y);
        return XMath.Exp(x) + XMath.Log(positive) + XMath.Sqrt(positive) + XMath.Square(x)
            + XMath.Abs(x) + XMath.Min(x, 1) + XMath.Max(x, -1) + XMath.Expm1(x) + XMath.Log1p(positive)
            + XMath.Logistic(x) + XMath.LogLogistic(x) + XMath.Logit(XMath.Logistic(x))
            + XMath.Softplus(x) + XMath.Log1mExp(-positive)
            + XMath.Sin(x) + XMath.Cos(x) + XMath.Tan(x) + XMath.Tanh(x) + XMath.Atan(x);
    }
}
