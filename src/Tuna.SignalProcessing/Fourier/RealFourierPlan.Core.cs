using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Tuna.SignalProcessing;

using Tuna.LinearAlgebra;

public sealed partial class RealFourierPlan
{
    private readonly FourierPlan complexPlan;

    // 偶数長の後処理用回転因子 ω_n^k (k ∈ [0, n/4])
    private readonly Complex[] twiddles;

    private const double PairWork = 64;

    private RealFourierPlan(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        Length = length;
        if (length > 0 && length % 2 == 0)
        {
            int h = length / 2;
            complexPlan = FourierPlan.Create(h);
            twiddles = new Complex[h / 2 + 1];
            for (int k = 0; k < twiddles.Length; k++)
            {
                twiddles[k] = UnitRoots.Root(k, length);
            }
        }
        else
        {
            complexPlan = FourierPlan.Create(length);
            twiddles = [];
        }
    }

    private bool IsEven => Length > 0 && Length % 2 == 0;

    internal int ForwardWorkLength(nint batch) =>
        !IsEven ? checked((int)(batch * Length) + complexPlan.WorkLength(batch))
        : batch == 1 ? complexPlan.WorkLength(1)
        : checked((int)(batch * (Length / 2)) + complexPlan.WorkLength(batch));

    internal int InverseWorkLength(nint batch) =>
        !IsEven ? checked((int)(batch * Length) + complexPlan.WorkLength(batch))
        : batch == 1 ? checked(Length / 2 + complexPlan.WorkLength(1))
        : checked((int)(2 * batch * (Length / 2)) + complexPlan.WorkLength(batch));

    internal unsafe void ForwardCore(ref double input, ref Complex output, Span<Complex> work, nint batch)
    {
        int n = Length;
        if (n == 0 || batch == 0)
        {
            return;
        }

        if (!IsEven)
        {
            int count = (int)(batch * n);
            var buffer = work[..count];
            for (int i = 0; i < count; i++)
            {
                buffer[i] = new Complex(Unsafe.Add(ref input, i), 0);
            }

            complexPlan.Execute<ForwardDirection>(ref buffer[0], ref buffer[0], work[count..], batch);
            buffer[..(int)(batch * SpectrumLength)].CopyTo(MemoryMarshal.CreateSpan(ref output, (int)(batch * SpectrumLength)));
            return;
        }

        // 偶数長: z_j = x_{2j} + i·x_{2j+1} の長さ h = n/2 の FFT から片側スペクトルを復元する。
        int h = n / 2;
        if (batch == 1)
        {
            // 1 本の場合は連続する実数対をそのまま複素数として再解釈できる。
            complexPlan.Execute<ForwardDirection>(ref Unsafe.As<double, Complex>(ref input), ref output, work, 1);
        }
        else
        {
            ref Complex z = ref work[0];
            fixed (double* source = &input)
            fixed (Complex* target = &z)
            {
                ForRange(0, h, batch * h, new Pack((nint)source, (nint)target, batch));
            }

            complexPlan.Execute<ForwardDirection>(ref z, ref output, work[(int)(batch * h)..], batch);
        }

        // X_0 = Re(Z_0) + Im(Z_0), X_h = Re(Z_0) - Im(Z_0)
        ref Complex last = ref Unsafe.Add(ref output, batch * h);
        for (nint q = 0; q < batch; q++)
        {
            Complex z0 = Unsafe.Add(ref output, q);
            Unsafe.Add(ref output, q) = new Complex(z0.Real + z0.Imaginary, 0);
            Unsafe.Add(ref last, q) = new Complex(z0.Real - z0.Imaginary, 0);
        }

        fixed (Complex* pointer = &output)
        {
            ForRange(1, h / 2 + 1, batch * h / 2, new ForwardPairs((nint)pointer, batch, h, twiddles));
        }
    }

    internal unsafe void InverseCore(ref Complex input, ref double output, Span<Complex> work, nint batch)
    {
        int n = Length;
        if (n == 0 || batch == 0)
        {
            return;
        }

        if (!IsEven)
        {
            int count = (int)(batch * n);
            var buffer = work[..count];
            for (nint q = 0; q < batch; q++)
            {
                buffer[(int)q] = new Complex(Unsafe.Add(ref input, q).Real, 0);
            }

            for (int k = 1; k <= (n - 1) / 2; k++)
            {
                for (nint q = 0; q < batch; q++)
                {
                    Complex x = Unsafe.Add(ref input, q + batch * k);
                    buffer[(int)(q + batch * k)] = x;
                    buffer[(int)(q + batch * (n - k))] = Complex.Conjugate(x);
                }
            }

            complexPlan.Execute<InverseDirection>(ref buffer[0], ref buffer[0], work[count..], batch);
            for (int i = 0; i < count; i++)
            {
                Unsafe.Add(ref output, i) = buffer[i].Real;
            }

            return;
        }

        // 偶数長: Z'_k = A + i·conj(ω_n^k)·D を長さ h で逆変換すると n·(x_{2j} + i·x_{2j+1}) が得られる。
        int h = n / 2;
        ref Complex z = ref work[0];
        Span<Complex> rest = work[(int)(batch * h)..];

        ref Complex last = ref Unsafe.Add(ref input, batch * h);
        for (nint q = 0; q < batch; q++)
        {
            double r0 = Unsafe.Add(ref input, q).Real;
            double rh = Unsafe.Add(ref last, q).Real;
            Unsafe.Add(ref z, q) = new Complex(r0 + rh, r0 - rh);
        }

        fixed (Complex* source = &input)
        fixed (Complex* target = &z)
        {
            ForRange(1, h / 2 + 1, batch * h / 2, new InversePairs((nint)source, (nint)target, batch, h, twiddles));
        }

        if (batch == 1)
        {
            complexPlan.Execute<InverseDirection>(ref z, ref Unsafe.As<double, Complex>(ref output), rest, 1);
            return;
        }

        ref Complex result = ref rest[0];
        complexPlan.Execute<InverseDirection>(ref z, ref result, rest[(int)(batch * h)..], batch);
        fixed (Complex* source = &result)
        fixed (double* target = &output)
        {
            ForRange(0, h, batch * h, new Unpack((nint)source, (nint)target, batch));
        }
    }

    private interface IRangeOperation
    {
        void Run(int first, int end);
    }

    private static void ForRange<TOperation>(int first, int end, nint elements, TOperation operation)
        where TOperation : struct, IRangeOperation
    {
        int threads = Parallelism.Degree(PairWork * elements, Parallelism.Grain);
        if (threads > 1 && end - first >= threads)
        {
            ForRangeParallel(first, end, threads, operation);
            return;
        }

        operation.Run(first, end);
    }

    // 逐次経路でクロージャーの割り当てを避けるため、並列実行を別メソッドに分離している。
    private static void ForRangeParallel<TOperation>(int first, int end, int threads, TOperation operation)
        where TOperation : struct, IRangeOperation =>
        Parallelism.ForStatic(threads, part => operation.Run(
            first + (int)((long)(end - first) * part / threads), first + (int)((long)(end - first) * (part + 1) / threads)));

    private readonly unsafe struct Pack(nint input, nint output, nint batch) : IRangeOperation
    {
        public void Run(int first, int end)
        {
            double* x = (double*)input;
            Complex* z = (Complex*)output;
            for (int j = first; j < end; j++)
            {
                double* even = x + batch * (2 * j);
                double* odd = x + batch * (2 * j + 1);
                Complex* target = z + batch * j;
                for (nint q = 0; q < batch; q++)
                {
                    target[q] = new Complex(even[q], odd[q]);
                }
            }
        }
    }

    private readonly unsafe struct Unpack(nint input, nint output, nint batch) : IRangeOperation
    {
        public void Run(int first, int end)
        {
            Complex* z = (Complex*)input;
            double* x = (double*)output;
            for (int j = first; j < end; j++)
            {
                Complex* source = z + batch * j;
                double* even = x + batch * (2 * j);
                double* odd = x + batch * (2 * j + 1);
                for (nint q = 0; q < batch; q++)
                {
                    even[q] = source[q].Real;
                    odd[q] = source[q].Imaginary;
                }
            }
        }
    }

    private readonly unsafe struct ForwardPairs(nint spectrum, nint batch, int h, Complex[] twiddles) : IRangeOperation
    {
        public void Run(int first, int end)
        {
            ref Complex z = ref Unsafe.AsRef<Complex>((void*)spectrum);
            int k = first;
            if (batch == 1)
            {
                if (Lanes4.IsAccelerated)
                {
                    k = Vector<Lanes4>(ref z, k, end);
                }
                else if (Lanes2.IsAccelerated)
                {
                    k = Vector<Lanes2>(ref z, k, end);
                }
            }

            for (; k < end; k++)
            {
                ref Complex column = ref Unsafe.Add(ref z, batch * k);
                LaneLoops.Apply(ref column, ref column, batch, new ForwardPair(twiddles[k], batch * (h - 2 * k)));
            }
        }

        // ペア (k, h - k) が重ならない範囲（k + Count - 1 < h - (k + Count - 1)）のみ SIMD で一括処理する。
        private int Vector<TL>(ref Complex z, int k, int end)
            where TL : unmanaged, IComplexLanes<TL>
        {
            ref Complex w = ref MemoryMarshal.GetArrayDataReference(twiddles);
            for (; k + TL.Count <= end && 2 * (k + TL.Count - 1) < h; k += TL.Count)
            {
                ref Complex first = ref Unsafe.Add(ref z, k);
                ref Complex partner = ref Unsafe.Add(ref z, h - k - (TL.Count - 1));
                TL zk = TL.Load(in first);
                TL zp = TL.Conjugate(TL.Reverse(TL.Load(in partner)));
                TL a = TL.Add(zk, zp);
                TL c = TL.MultiplyEach(TL.Subtract(zk, zp), TL.Load(in Unsafe.Add(ref w, k)));
                TL.Store(ref first, TL.Scale(TL.Add(a, TL.MultiplyMinusI(c)), 0.5));
                TL.Store(ref partner, TL.Reverse(TL.Scale(TL.Add(TL.Conjugate(a), TL.MultiplyMinusI(TL.Conjugate(c))), 0.5)));
            }

            return k;
        }
    }

    private readonly unsafe struct InversePairs(nint spectrum, nint output, nint batch, int h, Complex[] twiddles) : IRangeOperation
    {
        public void Run(int first, int end)
        {
            ref Complex x = ref Unsafe.AsRef<Complex>((void*)spectrum);
            ref Complex z = ref Unsafe.AsRef<Complex>((void*)output);
            int k = first;
            if (batch == 1)
            {
                if (Lanes4.IsAccelerated)
                {
                    k = Vector<Lanes4>(ref x, ref z, k, end);
                }
                else if (Lanes2.IsAccelerated)
                {
                    k = Vector<Lanes2>(ref x, ref z, k, end);
                }
            }

            for (; k < end; k++)
            {
                LaneLoops.Apply(
                    ref Unsafe.Add(ref x, batch * k), ref Unsafe.Add(ref z, batch * k), batch,
                    new InversePair(twiddles[k], batch * (h - 2 * k)));
            }
        }

        private int Vector<TL>(ref Complex x, ref Complex z, int k, int end)
            where TL : unmanaged, IComplexLanes<TL>
        {
            ref Complex w = ref MemoryMarshal.GetArrayDataReference(twiddles);
            for (; k + TL.Count <= end && 2 * (k + TL.Count - 1) < h; k += TL.Count)
            {
                int p = h - k - (TL.Count - 1);
                TL xk = TL.Load(in Unsafe.Add(ref x, k));
                TL xp = TL.Conjugate(TL.Reverse(TL.Load(in Unsafe.Add(ref x, p))));
                TL a = TL.Add(xk, xp);
                TL c = TL.MultiplyEachConjugate(TL.Subtract(xk, xp), TL.Load(in Unsafe.Add(ref w, k)));
                TL.Store(ref Unsafe.Add(ref z, k), TL.Add(a, TL.MultiplyI(c)));
                TL.Store(ref Unsafe.Add(ref z, p), TL.Reverse(TL.Add(TL.Conjugate(a), TL.MultiplyI(TL.Conjugate(c)))));
            }

            return k;
        }
    }

    private readonly struct ForwardPair(Complex w, nint partnerOffset) : ILaneOperation
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Invoke<TL>(ref Complex first, ref Complex second) where TL : unmanaged, IComplexLanes<TL>
        {
            ref Complex partner = ref Unsafe.Add(ref first, partnerOffset);
            TL zk = TL.Load(in first);
            TL zp = TL.Conjugate(TL.Load(in partner));
            TL a = TL.Add(zk, zp);
            TL c = TL.Multiply(TL.Subtract(zk, zp), w);
            TL.Store(ref first, TL.Scale(TL.Add(a, TL.MultiplyMinusI(c)), 0.5));
            if (partnerOffset != 0)
            {
                TL.Store(ref partner, TL.Scale(TL.Add(TL.Conjugate(a), TL.MultiplyMinusI(TL.Conjugate(c))), 0.5));
            }
        }
    }

    private readonly struct InversePair(Complex w, nint partnerOffset) : ILaneOperation
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Invoke<TL>(ref Complex first, ref Complex second) where TL : unmanaged, IComplexLanes<TL>
        {
            TL xk = TL.Load(in first);
            TL xp = TL.Conjugate(TL.Load(in Unsafe.Add(ref first, partnerOffset)));
            TL a = TL.Add(xk, xp);
            TL c = TL.MultiplyConjugate(TL.Subtract(xk, xp), w);
            TL.Store(ref second, TL.Add(a, TL.MultiplyI(c)));
            if (partnerOffset != 0)
            {
                TL.Store(ref Unsafe.Add(ref second, partnerOffset), TL.Add(TL.Conjugate(a), TL.MultiplyI(TL.Conjugate(c))));
            }
        }
    }
}
