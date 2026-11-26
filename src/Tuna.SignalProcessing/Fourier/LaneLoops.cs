using System.Numerics;
using System.Runtime.CompilerServices;

namespace Tuna.SignalProcessing;

internal interface ILaneOperation
{
    void Invoke<TL>(ref Complex first, ref Complex second) where TL : unmanaged, IComplexLanes<TL>;
}

internal static class LaneLoops
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Apply<TOperation>(ref Complex first, ref Complex second, nint count, TOperation operation)
        where TOperation : struct, ILaneOperation
    {
        nint q = 0;
        if (Lanes4.IsAccelerated)
        {
            for (; q <= count - 4; q += 4)
            {
                operation.Invoke<Lanes4>(ref Unsafe.Add(ref first, q), ref Unsafe.Add(ref second, q));
            }
        }

        if (Lanes2.IsAccelerated)
        {
            for (; q <= count - 2; q += 2)
            {
                operation.Invoke<Lanes2>(ref Unsafe.Add(ref first, q), ref Unsafe.Add(ref second, q));
            }
        }

        for (; q < count; q++)
        {
            operation.Invoke<Lanes1>(ref Unsafe.Add(ref first, q), ref Unsafe.Add(ref second, q));
        }
    }

    internal static void Scale(Span<double> data, double factor)
    {
        if (factor == 1)
        {
            return;
        }

        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated)
        {
            var f = new System.Numerics.Vector<double>(factor);
            var vectors = System.Runtime.InteropServices.MemoryMarshal.Cast<double, System.Numerics.Vector<double>>(data);
            for (int v = 0; v < vectors.Length; v++)
            {
                vectors[v] *= f;
            }

            i = vectors.Length * System.Numerics.Vector<double>.Count;
        }

        for (; i < data.Length; i++)
        {
            data[i] *= factor;
        }
    }

    internal static void Scale(Span<Complex> data, double factor) =>
        Scale(System.Runtime.InteropServices.MemoryMarshal.Cast<Complex, double>(data), factor);
}
