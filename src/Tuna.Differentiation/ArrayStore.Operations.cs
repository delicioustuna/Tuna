using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

internal sealed partial class ArrayStore<T>
{
    private ArrayView[] operationInputs = [];
    private int operationInputCount;
    private List<UserOperation>? operations;
    private Dictionary<Type, IOperationBuffer>? operationBuffers;
    private Dictionary<int, int>? operationSavedEnds;
    private T[]?[] broadcastOperationGradients = [];
    private int broadcastOperationGradientCount;

    public Span<ArrayView> ReserveOperationInputs(int length)
    {
        int required = checked(operationInputCount + length);
        if (required > operationInputs.Length)
        {
            Array.Resize(ref operationInputs, Math.Max(required, checked(operationInputs.Length * 2)));
        }

        return operationInputs.AsSpan(operationInputCount, length);
    }

    public int RecordOperation<TOperation>(TOperation operation, int inputCount)
        where TOperation : IDifferentiableOperation<T>
    {
        operationBuffers ??= new();
        if (!operationBuffers.TryGetValue(typeof(TOperation), out var buffer))
        {
            buffer = new OperationBuffer<TOperation>();
            operationBuffers.Add(typeof(TOperation), buffer);
        }

        int operationIndex = ((OperationBuffer<TOperation>)buffer).Add(operation);
        operations ??= new();
        int index = operations.Count;
        operations.Add(new(buffer, operationIndex, operationInputCount, inputCount));
        operationInputCount += inputCount;
        return index;
    }

    public Span<T> AllocateOperationSaved(int output, int length)
    {
        operationSavedEnds ??= new();
        int saved = AddPooled(length, 1, requiresGradient: false);
        operationSavedEnds[output] = saved + 1;
        return Values(saved);
    }

    public ReadOnlySpan<T> GetOperationSaved(int output, int index)
    {
        if (operationSavedEnds is null || !operationSavedEnds.TryGetValue(output, out int end) ||
            (uint)index >= (uint)(end - output - 1))
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        return Values(output + 1 + index);
    }

    public void DiscardOperationSlots(int output)
    {
        long limit = RetentionFactor * Math.Max(peakBorrowed, borrowed);
        DiscardOperationSlots(output, output);
        if (retained > limit)
        {
            // Forward failures can churn through array lengths within the same epoch.
            TrimSpares(limit, discardEmptySpares: true);
        }
    }

    public void DiscardOperationSavedSlots(int output)
    {
        int firstSaved = output + 1;
        if (firstSaved < count)
        {
            DiscardOperationSlots(output, firstSaved);
        }
    }

    private void DiscardOperationSlots(int output, int first)
    {
        // Forward can only append saved-value slots after its output slot.
        for (int slot = first; slot < count; slot++)
        {
            sources[slot] = null;
            borrowed -= Length(slot);
        }

        operationSavedEnds?.Remove(output);
        cachedCount = Math.Max(cachedCount, count);
        reviewRetention = true;
        count = first;
    }

    public MatrixSpan<T> OperationGradient(int index, scoped in ArrayView view) =>
        OperationGradient(index, view, allowOverwrite: false, out _);

    public MatrixSpan<T> OperationGradient(int index, scoped in ArrayView view, bool allowOverwrite, out bool overwrite)
    {
        if (view.Broadcast)
        {
            overwrite = false;
            if (index >= broadcastOperationGradients.Length)
            {
                Array.Resize(ref broadcastOperationGradients, index + 1);
            }

            T[]? buffer = broadcastOperationGradients[index];
            if (buffer is null)
            {
                buffer = Rent(view.Length);
                buffer.AsSpan().Clear();
                broadcastOperationGradients[index] = buffer;
                broadcastOperationGradientCount = Math.Max(broadcastOperationGradientCount, index + 1);
                borrowed += buffer.Length;
            }

            return new MatrixSpan<T>(buffer, view.Rows, view.Columns);
        }

        Span<T> gradient = BeginGradient(view, out bool fresh);
        if (fresh && !allowOverwrite)
        {
            gradient.Clear();
            overwrite = false;
        }
        else
        {
            overwrite = fresh;
        }

        return WritableMatrixOf(gradient, view);
    }

    private void ResetOperations()
    {
        operationInputCount = 0;
        operations?.Clear();
        operationSavedEnds?.Clear();
        if (operationBuffers is not null)
        {
            foreach (var buffer in operationBuffers.Values)
            {
                buffer.Reset();
            }
        }
    }

    private void UserOperationBackward(in ArrayRecord record)
    {
        int slot = record.Output;
        if (!Touched(slot))
        {
            return;
        }

        var operation = operations![record.ScalarInput];
        var views = operationInputs.AsSpan(operation.FirstInput, operation.InputCount);
        var context = new DifferentiableOperationContext<T>(this, views, slot, backward: true);
        var output = ArrayView.Whole(slot, rows[slot], columns[slot]);
        try
        {
            operation.Buffer.PullBack(operation.Index, context, MatrixView(output), GradientMatrix(output));
            for (int i = 0; i < broadcastOperationGradientCount; i++)
            {
                if (broadcastOperationGradients[i] is { } buffer)
                {
                    var view = views[i];
                    AddReduced(view, BeginGradient(view, out bool overwrite), buffer, overwrite, negate: false);
                }
            }
        }
        finally
        {
            ReturnOperationGradients();
        }
    }

    private void ReturnOperationGradients()
    {
        if (broadcastOperationGradientCount == 0)
        {
            return;
        }

        // Account for the peak while contributions and input adjoints coexist.
        peakBorrowed = Math.Max(peakBorrowed, borrowed);
        for (int i = 0; i < broadcastOperationGradientCount; i++)
        {
            if (broadcastOperationGradients[i] is { } buffer)
            {
                broadcastOperationGradients[i] = null;
                borrowed -= buffer.Length;
                Return(buffer);
            }
        }

        broadcastOperationGradientCount = 0;
    }

    private readonly record struct UserOperation(IOperationBuffer Buffer, int Index, int FirstInput, int InputCount);

    private interface IOperationBuffer
    {
        void PullBack(int index, DifferentiableOperationContext<T> context, ReadOnlyMatrixSpan<T> output, ReadOnlyMatrixSpan<T> gradient);
        void Reset();
    }

    // Store operations by their concrete type to avoid boxing value-type operations.
    private sealed class OperationBuffer<TOperation> : IOperationBuffer where TOperation : IDifferentiableOperation<T>
    {
        private readonly List<TOperation> items = [];

        public int Add(TOperation operation)
        {
            int index = items.Count;
            items.Add(operation);
            return index;
        }

        public void PullBack(int index, DifferentiableOperationContext<T> context, ReadOnlyMatrixSpan<T> output, ReadOnlyMatrixSpan<T> gradient)
            => items[index].PullBack(context, output, gradient);

        public void Reset() => items.Clear();
    }
}
