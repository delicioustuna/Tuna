using System.Buffers;
using System.Numerics;

namespace Tuna.Differentiation;

public sealed partial class Tape
{
    internal ScalarNode RecordPointwiseSum<T, TOperation>(
        int length, scoped ReadOnlySpan<PointwiseInput> inputs, TOperation operation)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TOperation : IPointwiseOperation<T>, allows ref struct
    {
        double value = RecordPointwise<T, TOperation>(length, inputs, operation, sum: true, out int index);
        return index < 0 ? value : new ScalarNode(value, this, index, generation);
    }

    internal VectorNode<T> RecordPointwiseVector<T, TOperation>(
        int length, scoped ReadOnlySpan<PointwiseInput> inputs, TOperation operation)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TOperation : IPointwiseOperation<T>, allows ref struct
    {
        RecordPointwise<T, TOperation>(length, inputs, operation, sum: false, out int slot);
        return new(this, ArrayView.Whole(slot, length, 1));
    }

    private double RecordPointwise<T, TOperation>(
        int length, scoped ReadOnlySpan<PointwiseInput> inputs, TOperation operation, bool sum, out int output)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TOperation : IPointwiseOperation<T>, allows ref struct
    {
        TapeElementTypes.Require<T>();
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        bool requiresGradient = false;
        foreach (var input in inputs)
        {
            requiresGradient |= input.RequiresGradient;
        }

        EnsureCanApply(requiresGradient);
        bool record = requiresGradient && length != 0;
        var store = sum && !record ? null : Store<T>();
        int[]? rented = record && inputs.Length > 64 ? ArrayPool<int>.Shared.Rent(inputs.Length) : null;
        Span<int> derivativeSlots = !record ? [] : rented is null ? stackalloc int[inputs.Length] : rented.AsSpan(0, inputs.Length);
        derivativeSlots.Fill(-1);
        try
        {
            double value = ForwardPointwise(store, length, inputs, operation, sum, derivativeSlots, out output);
            if (record)
            {
                int index = store!.RecordPointwise(inputs, derivativeSlots);
                if (sum)
                {
                    output = AppendScalarOutput();
                }

                var kind = sum ? ArrayOperation.PointwiseSum : ArrayOperation.PointwiseVector;
                AppendArrayRecord<T>(kind, output, default, default, 0, index);
            }

            return value;
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<int>.Shared.Return(rented);
            }
        }
    }

    private double ForwardPointwise<T, TOperation>(ArrayStore<T>? store, int length, scoped ReadOnlySpan<PointwiseInput> inputs,
        TOperation operation, bool sum, scoped Span<int> derivativeSlots, out int output)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TOperation : IPointwiseOperation<T>, allows ref struct
    {
        int firstSlot = store?.SlotCount ?? 0;
        RecordingState previousState = state;
        state = RecordingState.Forward;
        try
        {
            output = sum ? -1 : store!.AddPooled(length, 1, !derivativeSlots.IsEmpty);
            for (int i = 0; i < derivativeSlots.Length; i++)
            {
                PointwiseInput input = inputs[i];
                if (input.RequiresGradient)
                {
                    int size = input.DerivativeLength(length, sum);
                    derivativeSlots[i] = store!.AddPooled(size, 1, requiresGradient: false);
                }
            }

            Span<T> values = sum ? [] : store!.Values(output);
            return operation.Forward(new(sum, values, store, derivativeSlots));
        }
        catch
        {
            store?.DiscardOperationSlots(firstSlot);
            throw;
        }
        finally
        {
            state = previousState;
        }
    }
}

internal enum PointwiseInputKind : byte
{
    None,
    SharedScalar,
    SharedArray,
    PointVector,
    PointColumns,
}

internal readonly struct PointwiseInput
{
    internal readonly PointwiseInputKind Kind;
    internal readonly ArrayView View;
    internal readonly int ScalarIndex;

    private PointwiseInput(PointwiseInputKind kind, ArrayView view = default, int scalarIndex = 0)
    {
        Kind = kind;
        View = view;
        ScalarIndex = scalarIndex;
    }

    internal bool RequiresGradient => Kind != PointwiseInputKind.None;

    internal int DerivativeLength(int length, bool sum) => Kind switch
    {
        PointwiseInputKind.SharedScalar => sum ? 1 : length,
        PointwiseInputKind.SharedArray => sum ? View.Length : checked(View.Length * length),
        _ => View.Length,
    };

    internal static PointwiseInput Bind(ScalarNode node, scoped ref Tape? tape)
    {
        node.Validate();
        if (node.Tape is not { } owner)
        {
            return default;
        }

        IncludeTape(ref tape, owner);
        return new(PointwiseInputKind.SharedScalar, scalarIndex: node.Index);
    }

    internal static PointwiseInput Bind<T>(VectorNode<T> node, PointwiseInputKind kind, scoped ref Tape? tape)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        IncludeTape(ref tape, node.Owner());
        return node.RequiresGradient ? new(kind, node.View) : default;
    }

    internal static PointwiseInput Bind<T>(MatrixNode<T> node, PointwiseInputKind kind, scoped ref Tape? tape)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        IncludeTape(ref tape, node.Owner());
        return node.RequiresGradient ? new(kind, node.View) : default;
    }

    private static void IncludeTape(ref Tape? tape, Tape owner)
    {
        if (tape is not null && tape != owner)
        {
            throw new InvalidOperationException("The inputs must belong to the same tape.");
        }

        tape = owner;
    }
}

internal interface IPointwiseOperation<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    double Forward(scoped PointwiseOutput<T> output);
}

internal readonly ref struct PointwiseOutput<T>(
    bool sum, Span<T> values = default, ArrayStore<T>? store = null, ReadOnlySpan<int> derivativeSlots = default)
    where T : unmanaged, IFloatingPointIeee754<T>
{
    private readonly ReadOnlySpan<int> slots = derivativeSlots;

    internal bool Sum { get; } = sum;
    internal Span<T> Values { get; } = values;

    internal Span<T> Derivatives(int index) =>
        slots.IsEmpty || slots[index] < 0 ? [] : store!.Values(slots[index]);
}
