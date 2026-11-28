using System.Numerics;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Tests.Differentiation;

public class PointwiseOperationTests
{
    [Fact]
    public void SetNodesBindConstantsAndValidateShapesAndTapeOwnership()
    {
        Tape? owner = null;
        var zero = default(ScalarSetNode<double>).Bind(3, ref owner, "zero");
        Assert.Null(owner);
        Assert.Equal(0, zero[2]);
        Assert.False(zero.Input.RequiresGradient);
        ScalarSetNode<float> constant = 2.5f;
        Assert.Equal(2.5f, constant.Bind(0, ref owner, nameof(constant)).SharedValue);

        var tape = new Tape();
        ScalarSetNode<double> points = tape.Leaf(new LinearAlgebra.Vector<double>([2, 4, 8]));
        VectorSetNode<double> shared = tape.Constant(new LinearAlgebra.Vector<double>([1, 3]));
        VectorSetNode<double> columns = tape.Leaf(Matrix<double>.Generate(2, 3, static (i, j) => 10 * j + i));
        var perPoint = points.Bind(3, ref owner, nameof(points));
        Assert.Same(tape, owner);
        Assert.Null(perPoint.SharedValue);
        Assert.Equal(8, perPoint[2]);
        var sharedValues = shared.Bind(0, ref owner, nameof(shared));
        Assert.True(sharedValues.IsShared);
        Assert.False(sharedValues.Input.RequiresGradient);
        Assert.Equal(3, sharedValues.SharedValue[1]);
        var perColumn = columns.Bind(3, ref owner, nameof(columns));
        Assert.False(perColumn.IsShared);
        Assert.Equal([20d, 21d], perColumn[2].ToArray());

        Assert.Throws<ArgumentException>(nameof(points), () =>
        {
            _ = points.Bind(2, ref owner, nameof(points));
        });
        Assert.Throws<ArgumentException>(nameof(columns), () =>
        {
            _ = columns.Bind(2, ref owner, nameof(columns));
        });
        Assert.Throws<InvalidOperationException>(() =>
        {
            _ = default(VectorSetNode<double>).Bind(0, ref owner, "uninitialized");
        });

        ScalarSetNode<double> foreign = new Tape().Leaf(1);
        Assert.Throws<InvalidOperationException>(() =>
        {
            _ = foreign.Bind(3, ref owner, nameof(foreign));
        });
        tape.Reset();
        Assert.Throws<InvalidOperationException>(() =>
        {
            _ = points.Bind(3, ref owner, nameof(points));
        });
        Assert.Throws<InvalidOperationException>(() =>
        {
            _ = shared.Bind(0, ref owner, nameof(shared));
        });
        Assert.Throws<InvalidOperationException>(() =>
        {
            _ = columns.Bind(3, ref owner, nameof(columns));
        });
    }

    [Theory]
    [InlineData(false, 1)]
    [InlineData(false, 2)]
    [InlineData(true, 1)]
    [InlineData(true, 2)]
    public void MixedInputsPropagateToStridedAndSharedViews(bool pointwise, int rows)
    {
        CheckMixedInputs<float>(pointwise, rows);
        CheckMixedInputs<double>(pointwise, rows);
    }

    private static void CheckMixedInputs<T>(bool pointwise, int rows) where T : unmanaged, IFloatingPointIeee754<T>
    {
        const int length = 3;
        var tape = new Tape();
        double scalarValue = Math.BitIncrement(2d);
        var scalar = tape.Leaf(scalarValue);
        var vector = tape.Leaf(new LinearAlgebra.Vector<T>([T.Zero, T.CreateChecked(2), T.CreateChecked(4), T.CreateChecked(6), T.Zero]));
        var parameter = Parameter<T>.Vector(4);
        parameter.Value[1] = T.CreateChecked(3);
        parameter.Value[2] = T.CreateChecked(7);
        parameter.Gradient.Fill(T.CreateChecked(10));
        var sharedVector = tape.UseVector(parameter)[1..3];
        var matrix = tape.Leaf(Matrix<T>.Generate(3, 3, static (i, j) => T.CreateChecked(5 * i + j + 1)));
        var sharedMatrix = matrix.T[1..3, ..2];
        var pointMatrix = tape.Leaf(Matrix<T>.Generate(4, 3, static (i, j) => T.CreateChecked(5 * i + j + 1)));
        var pointColumns = pointMatrix.T[1..(1 + rows), 1..4];
        Tape? owner = tape;
        var s = ((ScalarSetNode<T>)scalar).Bind(length, ref owner, nameof(scalar));
        var x = ((ScalarSetNode<T>)vector[1..4]).Bind(length, ref owner, nameof(vector));
        var w = ((VectorSetNode<T>)sharedVector).Bind(length, ref owner, nameof(sharedVector));
        var m = PointwiseInput.Bind(sharedMatrix, PointwiseInputKind.SharedArray, ref owner);
        var c = ((VectorSetNode<T>)pointColumns).Bind(length, ref owner, nameof(pointColumns));
        var operation = new MixedLinearOperation<T>(s, x, w, sharedMatrix.Value, c);
        ReadOnlySpan<PointwiseInput> inputs = [s.Input, x.Input, w.Input, m, c.Input];
        Assert.Equal(T.CreateTruncating(scalarValue), s.SharedValue);
        Assert.Equal(2, w.Input.DerivativeLength(length, sum: true));
        Assert.Equal(4, m.DerivativeLength(length, sum: true));
        Assert.Equal(6, w.Input.DerivativeLength(length, sum: false));
        Assert.Equal(12, m.DerivativeLength(length, sum: false));

        int[] upstream = pointwise ? [2, 0, -4] : [2, 2, 2];
        int[] values = rows == 1 ? [59, 136, 231] : [75, 188, 339];
        if (pointwise)
        {
            var result = tape.RecordPointwiseVector<T, MixedLinearOperation<T>>(length, inputs, operation);
            Assert.Equal(values.Select(T.CreateChecked).ToArray(), result.Value.ToArray());
            var weights = tape.Constant(new LinearAlgebra.Vector<T>(upstream.Select(T.CreateChecked).ToArray()));
            tape.Backward(XMath.Dot(result, weights));
        }
        else
        {
            var result = tape.RecordPointwiseSum<T, MixedLinearOperation<T>>(length, inputs, operation);
            Assert.Equal(values.Sum(), result.Value);
            tape.Backward(2 * result);
        }

        int total = 0;
        var dx = tape.GetGradient(vector);
        var dc = tape.GetGradient(pointMatrix);
        for (int i = 0; i < length; i++)
        {
            int adjoint = (i + 1) * upstream[i];
            total += adjoint;
            Assert.Equal(T.CreateChecked(2 * adjoint), dx[i + 1]);
            Assert.Equal(T.Zero, dc[i + 1, 0]);
            Assert.Equal(T.CreateChecked(adjoint), dc[i + 1, 1]);
            Assert.Equal(T.CreateChecked(rows == 2 ? 2 * adjoint : 0), dc[i + 1, 2]);
        }

        Assert.Equal(T.Zero, dx[0]);
        Assert.Equal(T.Zero, dx[4]);
        Assert.Equal(T.Zero, dc[0, 1]);
        Assert.Equal(total, tape.GetGradient(scalar));
        Assert.Equal([T.CreateChecked(10), T.CreateChecked(10 + total), T.CreateChecked(10 + 3 * total), T.CreateChecked(10)], parameter.Gradient.ToArray());
        var dm = tape.GetGradient(matrix);
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                int expected = (i, j) switch { (1, 1) => total, (0, 2) => 5 * total, _ => 0 };
                Assert.Equal(T.CreateChecked(expected), dm[i, j]);
            }
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SharedBroadcastMatrixMasksZeroAdjointsAndReducesAliases(bool pointwise)
    {
        var tape = new Tape();
        var leaf = tape.Leaf(new LinearAlgebra.Vector<double>([1, 2, 3, 4]));
        var view = leaf[1..3].Broadcast(2).T;
        Tape? owner = tape;
        var input = PointwiseInput.Bind(view, PointwiseInputKind.SharedArray, ref owner);
        if (pointwise)
        {
            var result = tape.RecordPointwiseVector<double, BroadcastPartials>(2, [input], new());
            tape.Backward(XMath.Dot(result, tape.Constant(new LinearAlgebra.Vector<double>([0, 2]))));
        }
        else
        {
            tape.Backward(2 * tape.RecordPointwiseSum<double, BroadcastPartials>(2, [input], new()));
        }

        Assert.Equal([0d, 6d, 14d, 0d], tape.GetGradient(leaf).ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void UnreachedAndRoundedZeroAdjointsDoNotPropagateInfinitePartials(bool pointwise)
    {
        var tape = new Tape();
        var scalar = tape.Leaf(2);
        var leaf = tape.Leaf(new LinearAlgebra.Vector<float>([1, 2]));
        var shared = tape.Leaf(new LinearAlgebra.Vector<float>([3, 4]));
        var columns = tape.Leaf(new Matrix<float>(2, 2));
        Tape? owner = tape;
        var s = ((ScalarSetNode<float>)scalar).Bind(2, ref owner, nameof(scalar));
        var x = ((ScalarSetNode<float>)leaf).Bind(2, ref owner, nameof(leaf));
        var w = ((VectorSetNode<float>)shared).Bind(2, ref owner, nameof(shared));
        var c = ((VectorSetNode<float>)columns).Bind(2, ref owner, nameof(columns));
        ReadOnlySpan<PointwiseInput> inputs = [s.Input, x.Input, w.Input, c.Input];
        if (pointwise)
        {
            var unreachable = tape.RecordPointwiseVector<float, InfinitePartials>(2, inputs, new());
            var touched = tape.RecordPointwiseVector<float, InfinitePartials>(2, inputs, new());
            tape.Backward(XMath.Dot(touched, tape.Constant(new LinearAlgebra.Vector<float>([0f, 0f]))));
            Assert.False(tape.Store<float>().Touched(unreachable.View.Slot));
            Assert.True(tape.Store<float>().Touched(touched.View.Slot));
        }
        else
        {
            var result = tape.RecordPointwiseSum<float, InfinitePartials>(2, inputs, new());
            tape.Backward(double.Epsilon * result);
        }

        Assert.Equal(0, tape.GetGradient(scalar));
        Assert.Equal([0f, 0f], tape.GetGradient(leaf).ToArray());
        Assert.Equal([0f, 0f], tape.GetGradient(shared).ToArray());
        Assert.Equal(0f, tape.GetGradient(columns)[0, 0]);
        Assert.Equal(0f, tape.GetGradient(columns)[1, 1]);
    }

    [Fact]
    public void FailedForwardPreservesPreviouslyRecordedOperations()
    {
        var tape = new Tape();
        var leaf = tape.Leaf(new LinearAlgebra.Vector<double>([2, 3]));
        var scale = tape.Leaf(1);
        var old = RecordRetainedModel(tape, leaf, scale, pointwise: false, fail: false);
        Assert.Throws<ArithmeticException>(() => RecordRetainedModel(tape, leaf, scale, pointwise: false, fail: true));
        Assert.Throws<ArithmeticException>(() => RecordRetainedModel(tape, leaf, scale, pointwise: true, fail: true));
        var next = RecordRetainedModel(tape, leaf, scale, pointwise: true, fail: false);
        tape.Backward(old + next);
        Assert.Equal([2d, 2d], tape.GetGradient(leaf).ToArray());
    }

    internal static ScalarNode RecordRetainedModel<T>(Tape tape, VectorNode<T> leaf, ScalarNode scale, bool pointwise, bool fail)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        int count = leaf.Length;
        Tape? owner = tape;
        var scalar = ((ScalarSetNode<T>)scale).Bind(count, ref owner, nameof(scale));
        var points = ((ScalarSetNode<T>)leaf).Bind(count, ref owner, nameof(leaf));
        var shared = ((VectorSetNode<T>)leaf).Bind(count, ref owner, nameof(leaf));
        var operation = new RetainedModel<T>(scalar, points, shared, count, fail);
        ReadOnlySpan<PointwiseInput> inputs = [scalar.Input, points.Input, shared.Input];
        return pointwise
            ? XMath.Sum(tape.RecordPointwiseVector<T, RetainedModel<T>>(count, inputs, operation))
            : tape.RecordPointwiseSum<T, RetainedModel<T>>(count, inputs, operation);
    }

    private readonly ref struct RetainedModel<T>(BoundScalarSet<T> scalar, BoundScalarSet<T> points, BoundVectorSet<T> shared, int count, bool fail)
        : IPointwiseOperation<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        private readonly BoundScalarSet<T> factor = scalar;
        private readonly BoundScalarSet<T> inputs = points;
        private readonly BoundVectorSet<T> sharedValues = shared;

        public double Forward(scoped PointwiseOutput<T> output)
        {
            Span<T> ds = output.Derivatives(0), dx = output.Derivatives(1), dw = output.Derivatives(2);
            ds.Clear();
            dw.Clear();
            double sum = 0;
            for (int i = 0; i < count; i++)
            {
                T value = factor[i] * inputs[i] + sharedValues[i][i] - inputs[i];
                if (output.Sum)
                {
                    sum += double.CreateTruncating(value);
                }
                else
                {
                    output.Values[i] = value;
                }

                ds[output.Sum ? 0 : i] += inputs[i];
                dx[i] = factor[i] - T.One;
                dw[output.Sum ? i : i * count + i] = T.One;
                if (fail)
                {
                    throw new ArithmeticException();
                }
            }

            return sum;
        }
    }

    private readonly ref struct MixedLinearOperation<T>(BoundScalarSet<T> scalar, BoundScalarSet<T> points,
        BoundVectorSet<T> sharedVector, ReadOnlyMatrixSpan<T> sharedMatrix, BoundVectorSet<T> columns) : IPointwiseOperation<T>
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        private readonly BoundScalarSet<T> s = scalar;
        private readonly BoundScalarSet<T> x = points;
        private readonly BoundVectorSet<T> w = sharedVector;
        private readonly ReadOnlyMatrixSpan<T> m = sharedMatrix;
        private readonly BoundVectorSet<T> c = columns;

        public double Forward(scoped PointwiseOutput<T> output)
        {
            Span<T> ds = output.Derivatives(0), dx = output.Derivatives(1), dw = output.Derivatives(2);
            Span<T> dm = output.Derivatives(3), dc = output.Derivatives(4);
            ds.Clear();
            dw.Clear();
            dm.Clear();
            T two = T.CreateChecked(2), three = T.CreateChecked(3), five = T.CreateChecked(5);
            int rows = c.VectorLength;
            double sum = 0;
            for (int i = 0; i < dx.Length; i++)
            {
                T factor = T.CreateChecked(i + 1);
                T value = factor * (s[i] + two * x[i] + w[i][0] + three * w[i][1] + m[0, 1] + five * m[1, 0]
                    + c[i][0] + (rows == 2 ? two * c[i][1] : T.Zero));
                if (output.Sum)
                {
                    sum += double.CreateTruncating(value);
                }
                else
                {
                    output.Values[i] = value;
                }

                int sharedIndex = output.Sum ? 0 : i;
                ds[sharedIndex] += factor;
                dx[i] = two * factor;
                dw[2 * sharedIndex] += factor;
                dw[2 * sharedIndex + 1] += three * factor;
                dm[4 * sharedIndex + 2] += factor;
                dm[4 * sharedIndex + 1] += five * factor;
                dc[i * rows] = factor;
                if (rows == 2)
                {
                    dc[i * rows + 1] = two * factor;
                }
            }

            return sum;
        }
    }

    private readonly struct BroadcastPartials : IPointwiseOperation<double>
    {
        public double Forward(scoped PointwiseOutput<double> output)
        {
            Span<double> derivatives = output.Derivatives(0);
            if (!output.Sum)
            {
                output.Values.Fill(1);
                derivatives[..4].Fill(double.PositiveInfinity);
                derivatives = derivatives[4..];
            }

            ((ReadOnlySpan<double>)[1, 2, 3, 4]).CopyTo(derivatives);
            return 2;
        }
    }

    private readonly struct InfinitePartials : IPointwiseOperation<float>
    {
        public double Forward(scoped PointwiseOutput<float> output)
        {
            output.Values.Fill(1);
            for (int i = 0; i < 4; i++)
            {
                output.Derivatives(i).Fill(float.PositiveInfinity);
            }

            return 2;
        }
    }
}
