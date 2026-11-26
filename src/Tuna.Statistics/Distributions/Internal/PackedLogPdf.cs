using System.Buffers;
using Tuna.Differentiation;

namespace Tuna.Statistics.Distributions;

internal static class PackedLogPdf
{
    internal static int PackedLength(int order) => checked((int)((long)order * (order + 1L) / 2));

    internal static bool RequiresGradient(ReadOnlySpan<ScalarNode> inputs)
    {
        bool result = false;
        foreach (ScalarNode input in inputs)
        {
            result |= input.RequiresGradient;
        }

        return result;
    }

    internal static ScalarNode Record(double value, ReadOnlySpan<ScalarNode> first, ReadOnlySpan<double> firstGradient,
        ReadOnlySpan<ScalarNode> second, ReadOnlySpan<double> secondGradient,
        ReadOnlySpan<ScalarNode> third = default, ReadOnlySpan<double> thirdGradient = default)
    {
        int count = checked(first.Length + second.Length + third.Length);
        var inputs = ArrayPool<(ScalarNode Input, double Derivative)>.Shared.Rent(count);
        try
        {
            int index = 0;
            Add(first, firstGradient, inputs, ref index);
            Add(second, secondGradient, inputs, ref index);
            Add(third, thirdGradient, inputs, ref index);
            return ScalarNode.Record(value, inputs.AsSpan(0, count));
        }
        finally
        {
            Array.Clear(inputs, 0, count);
            ArrayPool<(ScalarNode Input, double Derivative)>.Shared.Return(inputs);
        }
    }

    private static void Add(ReadOnlySpan<ScalarNode> nodes, ReadOnlySpan<double> gradient,
        Span<(ScalarNode Input, double Derivative)> inputs, ref int index)
    {
        for (int i = 0; i < nodes.Length; i++)
        {
            inputs[index++] = (nodes[i], gradient[i]);
        }
    }

    internal static void SolveTranspose(ReadOnlySpan<double> lower, Span<double> rhs, Span<double> numerators = default)
    {
        int n = rhs.Length;
        for (int i = n - 1; i >= 0; i--)
        {
            int offset = i * n;
            double value = rhs[i];
            for (int j = i + 1; j < n; j++)
            {
                value -= lower[offset + j] * rhs[j];
            }

            if (!numerators.IsEmpty)
            {
                numerators[i] = value;
            }

            rhs[i] = value / lower[offset + i];
        }
    }

    internal static void Inverse(ReadOnlySpan<double> lower, int n, Span<double> inverse)
    {
        inverse.Clear();
        for (int i = 0; i < n; i++)
        {
            Span<double> column = inverse.Slice(i * n, n);
            column[i] = 1;
            MultivariateMath.SolveLowerInPlace(lower, n, column);
            SolveTranspose(lower, column);
        }
    }
}
