using System.Numerics;
using Tuna.LinearAlgebra;
using DoubleVector = Tuna.LinearAlgebra.Vector<double>;
using FloatVector = Tuna.LinearAlgebra.Vector<float>;

namespace Tuna.Differentiation.Tests;

internal static partial class CustomFunctions
{
    [ElementwiseOperation(Derivative = nameof(SoftsignDerivative))]
    public static double Softsign(double x) => x / (1 + Math.Abs(x));

    private static double SoftsignDerivative(double x, double value)
    {
        double t = 1 - Math.Abs(value);
        return t * t;
    }

    [ElementwiseOperation(Over = [nameof(y), nameof(x)], PartialDerivatives = [nameof(TimesAddY), nameof(TimesAddX)])]
    public static float TimesAdd(float x, float scale, float y) => x * scale + y;

    private static float TimesAddX(float x, float scale, float y, float result) => scale;

    private static float TimesAddY(float x, float scale, float y, float result) => 1;

    [ElementwiseOperation(Over = [nameof(x), nameof(y), nameof(weight)],
        PartialDerivatives = [nameof(BlendX), nameof(BlendY), nameof(BlendWeight)],
        OperatorName = "Blend", OperatorContainer = "Operators")]
    public static T Blend<T>(T x, T scale, T y, T weight) where T : unmanaged, INumberBase<T>
        => scale * (x + (y - x) * weight);

    private static U BlendX<U>(U x, U scale, U y, U weight, U result) where U : unmanaged, INumberBase<U>
        => scale * (U.One - weight);

    private static T BlendY<T>(T x, T scale, T y, T weight, T result) where T : unmanaged, INumberBase<T> => scale * weight;

    private static T BlendWeight<T>(T x, T scale, T y, T weight, T result) where T : unmanaged, INumberBase<T> => scale * (y - x);

    [ElementwiseOperation(Derivative = nameof(IgnoreDerivative))]
    public static double Ignore(double x) => 1;

    private static double IgnoreDerivative(double x, double result) => x == 0 ? 0 : throw new InvalidOperationException();
}

[Collection(nameof(Tuna.Tests.Differentiation.ParallelismCollection))]
public class GeneratedDifferentiationTests
{
    [Fact]
    public void UnaryScalarMatchesTheHandwrittenRule()
    {
        var tape = new Tape();
        var x = tape.Leaf(-0.7);
        ScalarNode actual = CustomFunctions.Softsign(x);
        double actualValue = actual.Value;
        tape.Backward(actual);
        long gradient = Bits(tape.GetGradient(x));

        tape.Reset();
        x = tape.Leaf(-0.7);
        double y = CustomFunctions.Softsign(x.Value);
        double t = 1 - Math.Abs(y);
        tape.Backward(ScalarNode.Record(y, [(x, t * t)]));
        Assert.Equal(Bits(y), Bits(actualValue));
        Assert.Equal(Bits(y), Bits(CustomFunctions.Softsign((ScalarNode)(-0.7)).Value));
        Assert.Equal(gradient, Bits(tape.GetGradient(x)));
    }

    [Fact]
    public void BinaryFloatRuleUsesOverOrderAndRoundsScalarValues()
    {
        var tape = new Tape();
        const double original = 1.00000004;
        var x = tape.Leaf(original);
        var y = tape.Leaf(0.375);
        var output = CustomFunctions.TimesAdd(x, 1.25f, y);
        Assert.Equal(Bits(CustomFunctions.TimesAdd((float)original, 1.25f, 0.375f)), Bits(output.Value));
        tape.Backward(output);
        Assert.Equal(Bits(1.25), Bits(tape.GetGradient(x)));
        Assert.Equal(Bits(1), Bits(tape.GetGradient(y)));

        tape.Reset();
        var vx = tape.Leaf(new FloatVector([0.25f, -1.5f]));
        var vy = tape.Leaf(new FloatVector([2, 3]));
        var result = CustomFunctions.TimesAdd(vx, 1.25f, vy);
        Assert.Equal(CustomFunctions.TimesAdd(vx.Value, 1.25f, vy.Value).ToArray(), result.Value.ToArray());
        tape.Backward(XMath.Sum(result));
        Assert.Equal([1.25f, 1.25f], tape.GetGradient(vx).ToArray());
        Assert.Equal([1f, 1f], tape.GetGradient(vy).ToArray());
    }

    [Fact]
    public void TernaryGenericRuleHandlesAliasedTransposedAndSlicedInputs()
    {
        var tape = new Tape();
        var x = tape.Leaf(Matrix<double>.FromColumnMajor(2, 3, [1, 2, 3, 4, 5, 6]));
        var view = x.T[1.., ..];
        var w = tape.Constant(Matrix<double>.FromColumnMajor(2, 2, [0.125, 0.25, 0.625, 0.75]));
        const double scale = 1.3;
        var y = CustomFunctions.Blend(view, scale, view, w);
        var expectedValues = CustomFunctions.Blend(view.Value, scale, view.Value, w.Value);
        for (int column = 0; column < y.Columns; column++)
        {
            for (int row = 0; row < y.Rows; row++)
            {
                Assert.Equal(Bits(expectedValues[row, column]), Bits(y.Value[row, column]));
            }
        }

        var seed = tape.Constant(new DoubleVector([0.7, -1.1]));
        tape.Backward(XMath.Sum(y * seed));

        var gradient = tape.GetGradient(x);
        for (int column = 0; column < x.Columns; column++)
        {
            for (int row = 0; row < x.Rows; row++)
            {
                double expected = column == 0 ? 0 :
                    seed.Value[row] * (scale * (1 - w.Value[column - 1, row])) +
                    seed.Value[row] * (scale * w.Value[column - 1, row]);
                Assert.Equal(Bits(expected), Bits(gradient[row, column]));
            }
        }

        tape.Reset();
        var scalar = tape.Leaf(2);
        var scalarResult = CustomFunctions.Blend(scalar, scale, scalar, 0.25);
        tape.Backward(scalarResult);
        Assert.Equal(Bits(scale * (1 - 0.25) + scale * 0.25), Bits(tape.GetGradient(scalar)));
    }

    [Fact]
    public void ConstantsAndZeroAdjointsDoNotEvaluateUnneededDerivatives()
    {
        var tape = new Tape();
        Assert.False(CustomFunctions.Ignore((ScalarNode)3).RequiresGradient);
        var constant = tape.Constant(new DoubleVector([3, 4]));
        Assert.False(CustomFunctions.Ignore(constant).RequiresGradient);

        var x = tape.Leaf(new DoubleVector([0, 3]));
        var y = CustomFunctions.Ignore(x);
        var weights = tape.Constant(new DoubleVector([double.PositiveInfinity, 0]));
        tape.Backward(XMath.Dot(y, weights));
        Assert.Equal([0d, 0d], tape.GetGradient(x).ToArray());
    }

    [Fact]
    public void GeneratedOperationsValidateInputs()
    {
        var tape = new Tape();
        var values = new DoubleVector([0.5, -3, 2]);
        var x = tape.Leaf(values);
        var other = new Tape().Leaf(values);
        Assert.Throws<InvalidOperationException>(() => CustomFunctions.Blend(x, 2.0, other, x));
        Assert.Throws<ArgumentException>(() => CustomFunctions.Blend(x, 2.0, x[..1], x));
        tape.Reset();
        Assert.Throws<InvalidOperationException>(() => CustomFunctions.Softsign(x));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void GeneratedOperationsReuseTheTapeWithoutAllocations(int degree)
    {
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            if (degree != 0)
            {
                LinearAlgebraSettings.MaxDegreeOfParallelism = degree;
            }

            var tape = new Tape();
            var values = new DoubleVector(4096);
            values.AsSpan().Fill(0.5);
            for (int i = 0; i < 32; i++)
            {
                Run(tape, values);
            }

            GC.Collect(0, GCCollectionMode.Forced, blocking: true);
            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 32; i++)
            {
                Run(tape, values);
            }

            Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }

        static void Run(Tape tape, DoubleVector values)
        {
            tape.Reset();
            var input = tape.Leaf(values);
            var weight = tape.Constant(values);
            tape.Backward(XMath.Sum(CustomFunctions.Softsign(CustomFunctions.Blend(input, 1.5, input, weight))));
        }
    }

    private static long Bits(double value) => BitConverter.DoubleToInt64Bits(value);
}
