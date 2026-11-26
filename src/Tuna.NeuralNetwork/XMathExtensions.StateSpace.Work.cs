using System.Buffers;
using System.Numerics;
using Tuna.LinearAlgebra;

namespace Tuna.NeuralNetwork;

public static partial class XMathExtensions
{
    private static void RequireScanDisjoint<T>(ReadOnlyMatrixSpan<T> input, MatrixSpan<T> state, MatrixSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        Guard.RequireNoOverlap(input, destination, allowExact: false, nameof(destination));
        Guard.RequireNoOverlap(input, state, allowExact: false, nameof(state));
    }

    private unsafe struct DepthwiseWork<T> : Split.IRangeWork where T : unmanaged, IFloatingPointIeee754<T>
    {
        internal T* X, Kernel, Bias, Output;
        internal int Channels, Taps, InputStride, KernelStride, OutputStride;

        public readonly void Invoke(int start, int end)
        {
            nint[]? rented = null;
            try
            {
                Span<nint> pointers = Taps <= 128 ? stackalloc nint[Taps]
                    : (rented = ArrayPool<nint>.Shared.Rent(Taps)).AsSpan(0, Taps);
                fixed (nint* taps = pointers)
                {
                    for (int t = start; t < end; t++)
                    {
                        for (int k = 0; k < Taps; k++)
                        {
                            int source = t - Taps + 1 + k;
                            taps[k] = source >= 0 ? (nint)(X + (nint)source * InputStride) : 0;
                        }

                        Kernels.DepthwiseTaps((T**)taps, Taps, Kernel, KernelStride, Bias, Output + (nint)t * OutputStride, Channels);
                    }
                }
            }
            finally
            {
                if (rented is not null)
                {
                    ArrayPool<nint>.Shared.Return(rented);
                }
            }
        }
    }

    private unsafe struct ScanForwardWork<T> : IParallelWork where T : unmanaged, IFloatingPointIeee754<T>
    {
        internal T* X, Delta, A, B, C, Skip, Y, H, History;
        internal int InputStride, DeltaStride, AStride, BStride, CStride, OutputStride, StateStride;
        internal int Channels, Length, States, Units, Groups, CheckpointInterval;
        internal bool Approximate;

        public readonly void Invoke(int group)
        {
            var (u0, u1) = Split.Range(Units, Groups, group);
            int first = u0 * ChannelUnit, last = Math.Min(Channels, u1 * ChannelUnit);
            int count = last - first;
            if (count <= 0)
            {
                return;
            }

            int localStateLength = checked(count * States);
            var work = ArrayPool<T>.Shared.Rent(checked(2 * localStateLength + count * Length));
            try
            {
                fixed (T* decay = work)
                {
                    T* local = decay + localStateLength, output = local + localStateLength;
                    for (int k = 0; k < States; k++)
                    {
                        new ReadOnlySpan<T>(H + (nint)k * StateStride + first, count)
                            .CopyTo(new Span<T>(local + (nint)k * count, count));
                    }

                    for (int t = 0; t < Length; t++)
                    {
                        Kernels.ScanStep(A, AStride, Delta + (nint)t * DeltaStride, X + (nint)t * InputStride,
                            B + (nint)t * BStride, C + (nint)t * CStride, Skip, local - first, count,
                            output + (nint)t * count - first, decay, first, last, States, Approximate);

                        if (History is not null && ((t + 1) % CheckpointInterval == 0 || t + 1 == Length))
                        {
                            T* checkpoint = History + (nint)(t / CheckpointInterval + 1) * Channels * States;
                            for (int k = 0; k < States; k++)
                            {
                                new ReadOnlySpan<T>(local + (nint)k * count, count)
                                    .CopyTo(new Span<T>(checkpoint + (nint)k * Channels + first, count));
                            }
                        }
                    }

                    for (int k = 0; k < States; k++)
                    {
                        new ReadOnlySpan<T>(local + (nint)k * count, count)
                            .CopyTo(new Span<T>(H + (nint)k * StateStride + first, count));
                    }

                    for (int t = 0; t < Length; t++)
                    {
                        new ReadOnlySpan<T>(output + (nint)t * count, count)
                            .CopyTo(new Span<T>(Y + (nint)t * OutputStride + first, count));
                    }
                }
            }
            finally
            {
                ArrayPool<T>.Shared.Return(work);
            }
        }
    }
}
