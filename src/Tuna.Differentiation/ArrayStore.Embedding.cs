using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

internal sealed partial class ArrayStore<T>
{
    internal int SaveEmbeddingIndices(ReadOnlySpan<int> indices)
    {
        // Use tape-owned storage while preserving every bit of an Int32 index for float tensors.
        int perElement = Unsafe.SizeOf<T>() / sizeof(int);
        int length = (indices.Length - 1) / perElement + 1;
        int slot = AddPooled(length, 1, requiresGradient: false);
        indices.CopyTo(MemoryMarshal.Cast<T, int>(Values(slot)));
        return slot;
    }

    private void EmbeddingBackward(in ArrayRecord record)
    {
        if (!Touched(record.Output))
        {
            return;
        }

        var output = ArrayView.Whole(record.Output, rows[record.Output], columns[record.Output]);
        var gradient = GradientMatrix(output);
        var indices = MemoryMarshal.Cast<T, int>(Values(record.ScalarInput));
        var table = record.Left;
        try
        {
            var destination = OperationGradient(0, table);
            for (int i = 0; i < gradient.Columns; i++)
            {
                var column = destination.Column(indices[i]);
                XMath.Add(column.AsReadOnly(), gradient.Column(i), column);
            }

            if (table.Broadcast)
            {
                AddReduced(table, BeginGradient(table, out bool overwrite), broadcastOperationGradients[0]!, overwrite, negate: false);
            }
        }
        finally
        {
            ReturnOperationGradients();
        }
    }
}
