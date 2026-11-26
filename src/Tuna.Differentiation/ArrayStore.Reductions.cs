using System.Numerics;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

internal sealed partial class ArrayStore<T>
{
    private void ReductionBackward(in ArrayRecord record, double[] adjoints)
    {
        ArrayOperation operation = record.Operation;
        ArrayView x = record.Left;
        if (operation <= ArrayOperation.LogSumExp)
        {
            VectorReductionBackward(operation, x, adjoints[record.Output]);
            return;
        }

        int output = record.Output;
        if (!Touched(output) || !RequiresGradient(x.Slot))
        {
            return;
        }

        if (operation >= ArrayOperation.ColumnSoftmax)
        {
            SoftmaxBackward(operation, x, output);
            return;
        }

        bool columns = operation is ArrayOperation.ColumnSums or ArrayOperation.ColumnSumsOfSquares
            or ArrayOperation.ColumnMeans or ArrayOperation.ColumnLogSumExp;
        ReadOnlySpan<T> d = gradients[output].AsSpan(0, columns ? x.Columns : x.Rows);
        switch (operation)
        {
            case ArrayOperation.ColumnSums:
            case ArrayOperation.RowSums:
                AddRepeated(x, d, columns);
                break;
            case ArrayOperation.ColumnSumsOfSquares:
            case ArrayOperation.RowSumsOfSquares:
                {
                    Span<T> twice = Scratch(ReducedValues, d.Length);
                    for (int i = 0; i < d.Length; i++)
                    {
                        twice[i] = d[i] + d[i];
                    }

                    AddHadamard(x, MatrixView(x), Repeat(twice, columns, x.Rows, x.Columns));
                    break;
                }

            case ArrayOperation.ColumnMeans:
            case ArrayOperation.RowMeans:
                {
                    Span<T> shares = Scratch(ReducedValues, d.Length);
                    int count = columns ? x.Rows : x.Columns;
                    for (int i = 0; i < d.Length; i++)
                    {
                        shares[i] = Share(d[i], count);
                    }

                    AddRepeated(x, shares, columns);
                    break;
                }

            default:
                {
                    var weights = new MatrixSpan<T>(Scratch(LeftDerivative, x.Length), x.Rows, x.Columns);
                    SoftmaxWeights(MatrixView(x), weights, columns);
                    AddHadamard(x, weights.AsReadOnly(), Repeat(d, columns, x.Rows, x.Columns));
                    break;
                }
        }
    }

    private void VectorReductionBackward(ArrayOperation operation, in ArrayView x, double adjoint)
    {
        T g = FromDouble(adjoint);
        if (T.IsZero(g) || !RequiresGradient(x.Slot))
        {
            return;
        }

        int n = x.RowCount;
        switch (operation)
        {
            case ArrayOperation.SumOfSquares:
                AddScaled(x, g + g, Values(x.Slot).Slice(Offset(x), n));
                break;
            case ArrayOperation.Mean:
                {
                    T share = Share(g, n);
                    Span<T> target = BeginGradient(x, out bool overwrite).Slice(Offset(x), n);
                    if (overwrite)
                    {
                        target.Fill(share);
                    }
                    else
                    {
                        XMath.Add(new ReadOnlyVectorSpan<T>(target), share, new VectorSpan<T>(target));
                    }

                    break;
                }

            default:
                {
                    Span<T> weights = Scratch(LeftDerivative, n);
                    SoftmaxWeights(MatrixView(x), new MatrixSpan<T>(weights, n, 1), columns: true);
                    AddScaled(x, g, weights);
                    break;
                }
        }
    }

    private void SoftmaxBackward(ArrayOperation operation, in ArrayView x, int output)
    {
        int rowCount = x.Rows, columnCount = x.Columns, n = x.Length;
        bool columns = operation is ArrayOperation.ColumnSoftmax or ArrayOperation.ColumnLogSoftmax;
        var y = new ReadOnlyMatrixSpan<T>(Values(output).Slice(0, n), rowCount, columnCount);
        var dy = new ReadOnlyMatrixSpan<T>(gradients[output].AsSpan(0, n), rowCount, columnCount);
        Span<T> sums = Scratch(ReducedValues, columns ? columnCount : rowCount);
        Span<T> scratch = Scratch(Contribution, n);
        var contribution = new MatrixSpan<T>(scratch, rowCount, columnCount);
        ReadOnlyMatrixSpan<T> repeated = Repeat(sums, columns, rowCount, columnCount);
        if (operation is ArrayOperation.ColumnSoftmax or ArrayOperation.RowSoftmax)
        {
            XMath.Hadamard(y, dy, contribution);
            Sums(contribution.AsReadOnly(), sums, columns);
            XMath.Subtract(dy, repeated, contribution);
            AddHadamard(x, y, contribution.AsReadOnly());
            return;
        }

        Sums(dy, sums, columns);
        XMath.Exp(y, contribution);
        XMath.Hadamard(contribution.AsReadOnly(), repeated, contribution);
        XMath.Subtract(dy, contribution.AsReadOnly(), contribution);
        AddGradient(x, scratch, negate: false);
    }

    // Non-finite slice maxima or NaN entries make every output weight in that slice NaN, so checking index 0 suffices.
    private static void SoftmaxWeights(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> weights, bool columns)
    {
        if (columns)
        {
            XMath.ColumnSoftmax(x, weights);
            for (int j = 0; x.Rows > 0 && j < x.Columns; j++)
            {
                if (T.IsNaN(weights[0, j]))
                {
                    LimitWeights(x.Column(j), weights.Column(j));
                }
            }
        }
        else
        {
            XMath.RowSoftmax(x, weights);
            for (int i = 0; x.Columns > 0 && i < x.Rows; i++)
            {
                if (T.IsNaN(weights[i, 0]))
                {
                    LimitWeights(x.Row(i), weights.Row(i));
                }
            }
        }
    }

    private static void LimitWeights(ReadOnlyVectorSpan<T> x, VectorSpan<T> weights)
    {
        int infinities = 0;
        for (int i = 0; i < x.Length; i++)
        {
            T value = x[i];
            if (T.IsNaN(value))
            {
                weights.Fill(T.NaN);
                return;
            }

            if (T.IsPositiveInfinity(value))
            {
                infinities++;
            }
        }

        T share = FromDouble(1.0 / (infinities > 0 ? infinities : x.Length));
        for (int i = 0; i < x.Length; i++)
        {
            weights[i] = infinities == 0 || T.IsPositiveInfinity(x[i]) ? share : T.Zero;
        }
    }

    // Divides in double before rounding to T so large n values that are not exact in float incur only a single rounding.
    private static T Share(T g, int n) => FromDouble(double.CreateTruncating(g) / n);

    private static void Sums(ReadOnlyMatrixSpan<T> x, Span<T> destination, bool columns)
    {
        if (columns)
        {
            XMath.ColumnSums(x, new VectorSpan<T>(destination));
        }
        else
        {
            XMath.RowSums(x, new VectorSpan<T>(destination));
        }
    }

    private static ReadOnlyMatrixSpan<T> Repeat(ReadOnlySpan<T> v, bool columns, int rows, int columnCount)
        => columns ? new ReadOnlyVectorSpan<T>(v).BroadcastRows(rows) : new ReadOnlyVectorSpan<T>(v).Broadcast(columnCount);

    private void AddRepeated(in ArrayView target, ReadOnlySpan<T> v, bool columns)
    {
        ReadOnlyMatrixSpan<T> repeated = Repeat(v, columns, target.Rows, target.Columns);
        Span<T> gradient = BeginGradient(target, out bool overwrite);
        if (target.Broadcast)
        {
            Span<T> contribution = Scratch(Contribution, target.Length);
            repeated.CopyTo(new MatrixSpan<T>(contribution, target.Rows, target.Columns));
            AddReduced(target, gradient, contribution, overwrite, negate: false);
            return;
        }

        MatrixSpan<T> view = WritableMatrixOf(gradient, target);
        if (overwrite)
        {
            repeated.CopyTo(view);
        }
        else
        {
            XMath.Add(view.AsReadOnly(), repeated, view);
        }
    }

    // right may alias the Contribution scratch buffer (overwritten in place by XMath.Hadamard).
    private void AddHadamard(in ArrayView target, ReadOnlyMatrixSpan<T> left, ReadOnlyMatrixSpan<T> right)
    {
        Span<T> gradient = BeginGradient(target, out bool overwrite);
        if (overwrite && !target.Broadcast)
        {
            XMath.Hadamard(left, right, WritableMatrixOf(gradient, target));
            return;
        }

        Span<T> contribution = Scratch(Contribution, target.Length);
        var product = new MatrixSpan<T>(contribution, target.Rows, target.Columns);
        XMath.Hadamard(left, right, product);
        if (target.Broadcast)
        {
            AddReduced(target, gradient, contribution, overwrite, negate: false);
            return;
        }

        MatrixSpan<T> view = WritableMatrixOf(gradient, target);
        XMath.Add(view.AsReadOnly(), product.AsReadOnly(), view);
    }
}
