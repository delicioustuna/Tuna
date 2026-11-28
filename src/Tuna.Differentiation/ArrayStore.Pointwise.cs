using System.Runtime.InteropServices;

namespace Tuna.Differentiation;

internal sealed partial class ArrayStore<T>
{
    private List<PointwiseRecord>? pointwiseOperations;
    private List<PointwiseDerivative>? pointwiseDerivatives;

    public int RecordPointwise(ReadOnlySpan<PointwiseInput> inputs, ReadOnlySpan<int> slots)
    {
        pointwiseOperations ??= new();
        pointwiseDerivatives ??= new();
        int start = pointwiseDerivatives.Count;
        for (int i = 0; i < slots.Length; i++)
        {
            if (slots[i] >= 0)
            {
                pointwiseDerivatives.Add(new(inputs[i], slots[i]));
            }
        }

        int index = pointwiseOperations.Count;
        pointwiseOperations.Add(new(start, pointwiseDerivatives.Count - start));
        return index;
    }

    private void PointwiseBackward(in ArrayRecord record, double[] adjoints)
    {
        bool sum = record.Operation == ArrayOperation.PointwiseSum;
        T scalarAdjoint = sum ? FromDouble(adjoints[record.Output]) : T.Zero;
        if (sum ? T.IsZero(scalarAdjoint) : !Touched(record.Output))
        {
            return;
        }

        ReadOnlySpan<T> vectorAdjoint = sum ? [] : OutputGradient(record.Output);
        var operation = pointwiseOperations![record.ScalarInput];
        var derivatives = CollectionsMarshal.AsSpan(pointwiseDerivatives).Slice(operation.Start, operation.Count);
        foreach (var derivative in derivatives)
        {
            ReadOnlySpan<T> local = Values(derivative.Slot);
            if (local.IsEmpty)
            {
                continue;
            }

            var input = derivative.Input;
            if (input.Kind == PointwiseInputKind.SharedScalar)
            {
                T contribution = sum ? (T.IsZero(local[0]) ? T.Zero : scalarAdjoint * local[0]) : MaskedSum(vectorAdjoint, local);
                adjoints[input.ScalarIndex] += double.CreateTruncating(contribution);
            }
            else if (sum)
            {
                AddMaskedScale(input.View, local, scalarAdjoint);
            }
            else if (input.Kind == PointwiseInputKind.SharedArray)
            {
                AddPointwiseShared(input.View, vectorAdjoint, local);
            }
            else if (input.Kind == PointwiseInputKind.PointVector || input.View.Rows == 1)
            {
                AddMaskedProduct(input.View, vectorAdjoint, local);
            }
            else
            {
                AddPointwiseColumns(input.View, vectorAdjoint, local);
            }
        }
    }

    private void AddPointwiseShared(in ArrayView target, ReadOnlySpan<T> dy, ReadOnlySpan<T> local)
    {
        int length = target.Length;
        Span<T> contribution = Scratch(Contribution, length);
        contribution.Clear();
        for (int i = 0; i < dy.Length; i++)
        {
            T weight = dy[i];
            if (!T.IsZero(weight))
            {
                ElementwiseKernels<T>.MaskedScale(local.Slice(i * length, length), weight, contribution, overwrite: false);
            }
        }

        AddGradient(target, contribution, negate: false);
    }

    private void AddPointwiseColumns(in ArrayView target, ReadOnlySpan<T> dy, ReadOnlySpan<T> local)
    {
        Span<T> gradient = BeginGradient(target, out bool overwrite);
        bool contiguous = IsContiguous(target);
        Span<T> destination = contiguous ? gradient.Slice(Offset(target), target.Length) : Scratch(Contribution, target.Length);
        int rowCount = target.Rows;
        for (int i = 0; i < dy.Length; i++)
        {
            int offset = i * rowCount;
            ElementwiseKernels<T>.MaskedScale(local.Slice(offset, rowCount), dy[i], destination.Slice(offset, rowCount), !contiguous || overwrite);
        }

        if (!contiguous)
        {
            Scatter(target, gradient, destination, overwrite);
        }
    }

    private readonly record struct PointwiseRecord(int Start, int Count);

    private readonly record struct PointwiseDerivative(PointwiseInput Input, int Slot);
}
