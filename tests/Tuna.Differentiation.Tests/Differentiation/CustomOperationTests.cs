using Tuna.LinearAlgebra;
using DoubleVector = Tuna.LinearAlgebra.Vector<double>;

namespace Tuna.Differentiation.Tests;

public class CustomOperationTests
{
    [Fact]
    public void SavedValuesFollowEachOperationAndMatchAnalyticAndNumericalGradients()
    {
        var tape = new Tape();
        var values = new DoubleVector([0.25, -0.75, 1.25]);
        var x = tape.Leaf(values);
        var operation = new SavedSine(1.5);
        var y = VectorNode<double>.Record(operation, [x], x.Length);
        var z = VectorNode<double>.Record(operation, [2.0 * x], x.Length);
        tape.Backward(XMath.Sum(y + z));
        var gradient = tape.GetGradient(x);
        const double step = 1e-5;
        for (int i = 0; i < x.Length; i++)
        {
            double value = values[i];
            double analytic = 1.5 * (Math.Cos(value) + 2 * Math.Cos(2 * value));
            double difference = (Objective(value + step) - Objective(value - step)) / (2 * step);
            Assert.Equal(analytic, gradient[i], 14);
            Assert.InRange(Math.Abs(difference - gradient[i]), 0, 1e-9);
        }

        static double Objective(double x) => 1.5 * (Math.Sin(x) + Math.Sin(2 * x));
    }

    [Fact]
    public void ForwardFailureDiscardsEverySavedBufferAndOutput()
    {
        var tape = new Tape();
        var x = tape.Leaf(new DoubleVector([0.2, -0.8]));
        Assert.Throws<ArithmeticException>(() => tape.Apply(new SavedSine(9, FailForward: true), [x], x.Length));
        var y = tape.Apply(new SavedSine(2), [x], x.Length);
        var loss = XMath.Sum(y);
        tape.Backward(loss);
        Assert.Equal([2 * Math.Cos(0.2), 2 * Math.Cos(-0.8)], tape.GetGradient(x).ToArray());
    }

    [Fact]
    public void CallbacksAllowReadsAndRejectTapeMutations()
    {
        var tape = new Tape();
        var values = new DoubleVector([3]);
        var x = tape.Leaf(values);
        var constant = tape.Constant(values);
        var operation = new InspectOperation(() =>
        {
            Assert.Equal(3, x.Value[0]);
            Assert.True(x.RequiresGradient);
            Assert.Equal(3, constant.AsColumnMatrix().Value[0, 0]);
            Assert.Throws<InvalidOperationException>(() => tape.Constant(values));
            Assert.Throws<InvalidOperationException>(() => XMath.Square(constant));
            Assert.Throws<InvalidOperationException>(() => tape.Leaf(0));
        });

        var y = VectorNode<double>.Record(operation, [x], 1);
        tape.Backward(XMath.Sum(y));
        Assert.Equal(1, tape.GetGradient(x)[0]);
    }

    [Fact]
    public void FailedBackwardPreservesParameterContributionsAndReportsResetRequirement()
    {
        var tape = new Tape();
        var parameter = Parameter<double>.Vector(2);
        var parameters = new ParameterSet<double>(parameter);
        var x = tape.UseVector(parameter);
        var failed = tape.Apply(new FailingOperation(tape, failForward: false), [x], 1);
        ScalarNode loss = XMath.Sum(failed) + XMath.Sum(x * 3);

        Assert.Throws<InvalidOperationException>(() => tape.Backward(loss));
        Assert.Equal([3d, 3d], parameter.Gradient.ToArray());
        Assert.Contains("Backpropagation failed", Assert.Throws<InvalidOperationException>(() => tape.GetGradient(x)).Message);
        Assert.Contains("Call Reset", Assert.Throws<InvalidOperationException>(() => tape.GetGradient(loss)).Message);
        Assert.Contains("Backpropagation failed", Assert.Throws<InvalidOperationException>(() => tape.Backward(loss)).Message);
        Assert.Contains("Backpropagation failed", Assert.Throws<InvalidOperationException>(() => tape.Leaf(0)).Message);

        tape.Reset();
        parameters.ZeroGradients();
        tape.Backward(XMath.Sum(tape.UseVector(parameter) * 2));
        Assert.Equal([2d, 2d], parameter.Gradient.ToArray());
    }

    [Fact]
    public void ApplyReducesBothBroadcastAxesIntoTheSourceGradient()
    {
        var tape = new Tape();
        var x = tape.Leaf(new DoubleVector([2, 3]));
        var output = tape.Apply(new SumInputs(), [x.Broadcast(2), x.BroadcastRows(3)], 1, 1);
        Assert.Equal(25, output.Value[0, 0]);
        tape.Backward(XMath.Sum(XMath.ColumnSums(output)));
        Assert.Equal([5d, 5d], tape.GetGradient(x).ToArray());
    }

    [Fact]
    public void ApplyCanChangeShapeAndAccumulateIntoAliasedInputViews()
    {
        var tape = new Tape();
        var parameter = Parameter<double>.Matrix(2, 3);
        new double[] { 1, 2, 3, 4, 5, 6 }.CopyTo(parameter.Value);
        parameter.Gradient.Fill(2);
        var x = tape.UseMatrix(parameter);
        var seed = tape.Constant(new DoubleVector([1]));
        var y = tape.Apply(new SumInputs(), [x.T, x[.., 1..], seed.AsColumnMatrix()], 1, 1);
        Assert.Equal(40, y.Value[0, 0]);

        tape.Backward(XMath.Sum(y * seed));
        Assert.Equal([3d, 3d, 4d, 4d, 4d, 4d], parameter.Gradient.ToArray());
    }

    [Fact]
    public void ApplyRunsOnlyForwardForConstantsAndSkipsDisconnectedPullBack()
    {
        var tape = new Tape();
        var counts = new CallCounts();
        var x = tape.Constant(new DoubleVector([1, 2]));
        var y = tape.Apply(new SumInputs(counts), [x], 1);
        Assert.False(y.RequiresGradient);
        Assert.Equal(3, y.Value[0]);

        var leaf = tape.Leaf(new DoubleVector([3, 4]));
        _ = tape.Apply(new SumInputs(counts), [leaf], 1);
        tape.Backward(XMath.Sum(leaf));
        Assert.Equal(2, counts.Forward);
        Assert.Equal(0, counts.Backward);
        Assert.Equal([1d, 1d], tape.GetGradient(leaf).ToArray());
    }

    [Fact]
    public void CallbackFailuresLeaveForwardReusableAndRequireResetAfterBackward()
    {
        var tape = new Tape();
        var values = new DoubleVector([1, 2]);
        var x = tape.Leaf(values);
        Assert.Throws<InvalidOperationException>(() => tape.Apply(new FailingOperation(tape, failForward: true), [x], 1));
        var y = tape.Apply(new FailingOperation(tape, failForward: false), [x], 1);
        ScalarNode loss = XMath.Sum(y);
        Assert.Throws<InvalidOperationException>(() => tape.Backward(loss));
        Assert.Throws<InvalidOperationException>(() => tape.Backward(loss));
        Assert.Throws<InvalidOperationException>(() => tape.GetGradient(x));

        tape.Reset();
        x = tape.Leaf(values);
        y = tape.Apply(new SumInputs(), [x], 1);
        tape.Backward(XMath.Sum(y));
        Assert.Equal([1d, 1d], tape.GetGradient(x).ToArray());
    }

    private sealed class CallCounts
    {
        public int Forward;
        public int Backward;
    }

    internal static void RecordBroadcastSine(Tape tape, DoubleVector values, int columns, int failures = 0)
    {
        tape.Reset();
        var x = tape.Leaf(values);
        var input = x.Broadcast(columns);
        for (int i = 0; i < failures; i++)
        {
            AssertForwardFailure(tape, input);
        }

        var y = MatrixNode<double>.Record(new SavedSine(2), [input], x.Length, columns);
        tape.Backward(XMath.Sum(XMath.ColumnSums(y)));
    }

    private static void AssertForwardFailure(Tape tape, MatrixNode<double> input)
        => Assert.Throws<ArithmeticException>(() => tape.Apply(new SavedSine(2, FailForward: true), [input], input.Rows, input.Columns));

    private readonly record struct InspectOperation(Action Inspect) : IDifferentiableOperation<double>
    {
        public void Forward(DifferentiableOperationContext<double> context, MatrixSpan<double> destination)
        {
            Inspect();
            context[0].CopyTo(destination);
        }

        public void PullBack(DifferentiableOperationContext<double> context, ReadOnlyMatrixSpan<double> output, ReadOnlyMatrixSpan<double> outputGradient)
        {
            Inspect();
            var gradient = context.GetGradient(0);
            gradient[0, 0] += outputGradient[0, 0];
        }
    }

    private readonly record struct SavedSine(double Scale, bool FailForward = false) : IDifferentiableOperation<double>
    {
        public void Forward(DifferentiableOperationContext<double> context, MatrixSpan<double> destination)
        {
            var x = context[0];
            Span<double> derivatives = context.AllocateSaved(x.Rows * x.Columns);
            context.AllocateSaved(1)[0] = Scale;
            if (FailForward)
            {
                throw new ArithmeticException("Forward failed after saving.");
            }

            for (int column = 0; column < x.Columns; column++)
            {
                for (int row = 0; row < x.Rows; row++)
                {
                    double value = x[row, column];
                    destination[row, column] = Scale * Math.Sin(value);
                    derivatives[column * x.Rows + row] = Math.Cos(value);
                }
            }
        }

        public void PullBack(DifferentiableOperationContext<double> context, ReadOnlyMatrixSpan<double> output, ReadOnlyMatrixSpan<double> outputGradient)
        {
            var derivatives = context.GetSaved(0);
            double scale = context.GetSaved(1)[0];
            var gradient = context.GetGradient(0);
            for (int column = 0; column < gradient.Columns; column++)
            {
                for (int row = 0; row < gradient.Rows; row++)
                {
                    gradient[row, column] += scale * derivatives[column * gradient.Rows + row] * outputGradient[row, column];
                }
            }
        }
    }

    private readonly struct SumInputs(CallCounts? counts = null) : IDifferentiableOperation<double>
    {
        public void Forward(DifferentiableOperationContext<double> inputs, MatrixSpan<double> destination)
        {
            if (counts is not null)
            {
                counts.Forward++;
            }

            double sum = 0;
            for (int index = 0; index < inputs.Count; index++)
            {
                var value = inputs[index];
                for (int column = 0; column < value.Columns; column++)
                {
                    for (int row = 0; row < value.Rows; row++)
                    {
                        sum += value[row, column];
                    }
                }
            }

            destination[0, 0] = sum;
        }

        public void PullBack(DifferentiableOperationContext<double> inputs, ReadOnlyMatrixSpan<double> output, ReadOnlyMatrixSpan<double> outputGradient)
        {
            if (counts is not null)
            {
                counts.Backward++;
            }

            for (int index = 0; index < inputs.Count; index++)
            {
                if (inputs.RequiresGradient(index))
                {
                    var gradient = inputs.GetGradient(index);
                    XMath.Add(gradient.AsReadOnly(), outputGradient[0, 0], gradient);
                }
            }
        }
    }

    private readonly struct FailingOperation(Tape owner, bool failForward) : IDifferentiableOperation<double>
    {
        public void Forward(DifferentiableOperationContext<double> inputs, MatrixSpan<double> destination)
        {
            if (failForward)
            {
                owner.Reset();
            }

            destination[0, 0] = inputs[0][0, 0];
        }

        public void PullBack(DifferentiableOperationContext<double> inputs, ReadOnlyMatrixSpan<double> output, ReadOnlyMatrixSpan<double> outputGradient)
            => owner.Reset();
    }
}
