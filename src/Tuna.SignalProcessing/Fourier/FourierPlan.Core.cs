using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Tuna.SignalProcessing;

using Tuna.LinearAlgebra;

internal sealed class StockhamStage
{
    internal StockhamStage(int radix, int m, Complex[] twiddles, Complex[]? firstTwiddles, double[]? cos, double[]? sin)
    {
        Radix = radix;
        M = m;
        Twiddles = twiddles;
        FirstTwiddles = firstTwiddles;
        Cos = cos;
        Sin = sin;
    }

    internal int Radix { get; }

    internal int M { get; }

    // ω_N^(j1·k2), index = (j1 - 1)·(p - 1) + (k2 - 1) (j1 ≥ 1, k2 ≥ 1). Empty when FirstTwiddles is used.
    internal Complex[] Twiddles { get; }

    // 第 1 段 SIMD 用に j1 を 4 要素単位でパックした回転因子。
    internal Complex[]? FirstTwiddles { get; }

    internal double[]? Cos { get; }

    internal double[]? Sin { get; }
}

public sealed partial class FourierPlan
{
    internal const int LargestDirectPrime = 31;

    private const double StageWork = 32;

    private readonly StockhamStage[] stages;

    private readonly FourierPlan? convolutionPlan;
    private readonly Complex[]? chirp;
    private readonly Complex[]? kernelSpectrum;

    private FourierPlan(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        Length = length;
        var radices = length <= 1 ? [] : Factor(length);
        if (radices is not null)
        {
            stages = CreateStages(length, radices);
            return;
        }

        // Bluestein 法: jk = (j² + k² − (k − j)²)/2 より巡回畳み込みへ帰着する。
        stages = [];
        int m = NextFastLength(2 * length - 1);
        convolutionPlan = Create(m);
        chirp = new Complex[length];
        long twoN = 2L * length;
        for (int t = 0; t < length; t++)
        {
            // t² を 2n で剰余を取ってから評価し、角度の丸め誤差増大を防ぐ。
            chirp[t] = UnitRoots.Root((long)t * t % twoN, twoN);
        }

        var kernel = new Complex[m];
        kernel[0] = Complex.Conjugate(chirp[0]);
        for (int t = 1; t < length; t++)
        {
            kernel[t] = Complex.Conjugate(chirp[t]);
            kernel[m - t] = Complex.Conjugate(chirp[t]);
        }

        kernelSpectrum = new Complex[m];
        var work = new Complex[convolutionPlan.WorkLength(1)];
        convolutionPlan.Execute<ForwardDirection>(ref kernel[0], ref kernelSpectrum[0], work, 1);

        // 逆 FFT の 1/m スケーリングを事前に畳み込み核へ織り込む。
        double scale = 1.0 / m;
        for (int k = 0; k < m; k++)
        {
            kernelSpectrum[k] = new Complex(kernelSpectrum[k].Real * scale, kernelSpectrum[k].Imaginary * scale);
        }
    }

    internal bool UsesBluestein => convolutionPlan is not null;

    internal IEnumerable<int> Radices => (convolutionPlan?.stages ?? stages).Select(stage => stage.Radix);

    internal static int NextFastLength(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        if (length <= 1)
        {
            return 1;
        }

        // 2 の累乗候補が 2·length 未満に必ず存在するため、探索上限は 2·length で十分。
        long limit = 2L * length;
        long best = long.MaxValue;
        for (long p7 = 1; p7 < limit; p7 *= 7)
        {
            for (long p5 = p7; p5 < limit; p5 *= 5)
            {
                for (long p3 = p5; p3 < limit; p3 *= 3)
                {
                    long candidate = p3;
                    while (candidate < length)
                    {
                        candidate *= 2;
                    }

                    best = Math.Min(best, candidate);
                }
            }
        }

        if (best > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(length), "The length is too large.");
        }

        return (int)best;
    }

    private static List<int>? Factor(int n)
    {
        var radices = new List<int>();
        int twos = BitOperations.TrailingZeroCount(n);
        n >>= twos;

        // 2^twos = 8^a·4^b·2^c。余り 1 は 8·2 ではなく 4·4 に分解して基数 2 の段を避ける。
        int eights = twos / 3;
        int fours = 0;
        int rest = twos % 3;
        if (rest == 2)
        {
            fours = 1;
        }
        else if (rest == 1 && eights > 0)
        {
            eights--;
            fours = 2;
        }

        for (int i = 0; i < eights; i++)
        {
            radices.Add(8);
        }

        for (int i = 0; i < fours; i++)
        {
            radices.Add(4);
        }

        if (twos == 1)
        {
            radices.Add(2);
        }

        for (int p = 3; p <= LargestDirectPrime && n > 1; p += 2)
        {
            while (n % p == 0)
            {
                radices.Add(p);
                n /= p;
            }
        }

        return n == 1 ? radices : null;
    }

    private static StockhamStage[] CreateStages(int n, List<int> radices)
    {
        var result = new StockhamStage[radices.Count];
        long before = 1;
        for (int i = 0; i < radices.Count; i++)
        {
            int p = radices[i];
            int m = (int)(n / before / p);

            Complex[] twiddles = [];
            Complex[]? first = null;
            if (i == 0 && StockhamKernels.HasKernel(p))
            {
                first = new Complex[StockhamKernels.FirstLength(m, p)];
                for (int j1 = 0; j1 < m; j1++)
                {
                    int t = StockhamKernels.FirstIndex(j1, p);
                    for (int k2 = 1; k2 < p; k2++)
                    {
                        first[t + StockhamKernels.FirstBlock * (k2 - 1)] = UnitRoots.Root((long)j1 * k2, n);
                    }
                }
            }
            else
            {
                twiddles = new Complex[(m - 1) * (p - 1)];
                for (int j1 = 1; j1 < m; j1++)
                {
                    for (int k2 = 1; k2 < p; k2++)
                    {
                        twiddles[(j1 - 1) * (p - 1) + (k2 - 1)] = UnitRoots.Root(before * j1 * k2, n);
                    }
                }
            }

            double[]? cos = null, sin = null;
            if (!StockhamKernels.HasKernel(p))
            {
                cos = new double[p];
                sin = new double[p];
                for (int r = 0; r < p; r++)
                {
                    var root = UnitRoots.Root(r, p);
                    cos[r] = root.Real;
                    sin[r] = -root.Imaginary;
                }
            }

            result[i] = new StockhamStage(p, m, twiddles, first, cos, sin);
            before *= p;
        }

        return result;
    }

    internal int WorkLength(nint batch)
    {
        if (Length <= 1)
        {
            return 0;
        }

        long length = convolutionPlan is null ? (long)batch * Length : 2L * batch * convolutionPlan.Length;
        if (length > Array.MaxLength)
        {
            throw new ArgumentException("The transform is too large for a single work buffer.");
        }

        return (int)length;
    }

    internal void Execute<TD>(ref Complex input, ref Complex output, Span<Complex> work, nint batch)
        where TD : IFourierDirection
    {
        if (Length <= 1 || batch == 0)
        {
            if (!Unsafe.AreSame(ref input, ref output))
            {
                MemoryMarshal.CreateReadOnlySpan(ref input, (int)(batch * Length))
                    .CopyTo(MemoryMarshal.CreateSpan(ref output, (int)(batch * Length)));
            }

            return;
        }

        if (convolutionPlan is not null)
        {
            ExecuteBluestein<TD>(ref input, ref output, work, batch);
            return;
        }

        ref Complex buffer = ref MemoryMarshal.GetReference(work);
        if (stages.Length % 2 == 0)
        {
            // 段数が偶数のときは work から交互に書き出すため、インプレースでも退避不要。
            RunStages<TD>(ref input, ref buffer, ref output, batch);
        }
        else if (Unsafe.AreSame(ref input, ref output))
        {
            // 段数が奇数のインプレースでは最終出力先を output に合わせるため、事前に work へコピーする。
            int count = (int)(batch * Length);
            MemoryMarshal.CreateReadOnlySpan(ref input, count).CopyTo(work);
            RunStages<TD>(ref buffer, ref output, ref buffer, batch);
        }
        else
        {
            RunStages<TD>(ref input, ref output, ref buffer, batch);
        }
    }

    private bool RunStages<TD>(ref Complex input, ref Complex x, ref Complex y, nint batch)
        where TD : IFourierDirection
    {
        int threads = Parallelism.Degree(StageWork * (double)batch * Length, Parallelism.Grain);
        if (threads > 1)
        {
            RunStagesParallel<TD>(ref input, ref x, ref y, batch, threads);
            return stages.Length % 2 == 1;
        }

        nint stride = batch;
        ref Complex source = ref input;
        for (int i = 0; i < stages.Length; i++)
        {
            ref Complex destination = ref (i % 2 == 0 ? ref x : ref y);
            StockhamKernels.Run<TD>(stages[i], ref source, ref destination, stride);
            source = ref destination;
            stride *= stages[i].Radix;
        }

        return stages.Length % 2 == 1;
    }

    // 逐次経路でクロージャーの割り当てを避けるため、並列実行を別メソッドに分離している。
    private unsafe void RunStagesParallel<TD>(ref Complex input, ref Complex x, ref Complex y, nint batch, int threads)
        where TD : IFourierDirection
    {
        fixed (Complex* inputPointer = &input, xPointer = &x, yPointer = &y)
        {
            nint source = (nint)inputPointer;
            nint stride = batch;
            for (int i = 0; i < stages.Length; i++)
            {
                var stage = stages[i];
                nint from = source, to = i % 2 == 0 ? (nint)xPointer : (nint)yPointer, s = stride;
                Parallelism.ForStatic(threads, part => StockhamKernels.Run<TD>(
                    stage, ref Unsafe.AsRef<Complex>((void*)from), ref Unsafe.AsRef<Complex>((void*)to), s, part, threads));
                source = to;
                stride *= stage.Radix;
            }
        }
    }

    private void ExecuteBluestein<TD>(ref Complex input, ref Complex output, Span<Complex> work, nint batch)
        where TD : IFourierDirection
    {
        var inner = convolutionPlan!;
        int n = Length;
        int m = inner.Length;
        ref Complex u = ref MemoryMarshal.GetReference(work);
        ref Complex v = ref Unsafe.Add(ref u, batch * m);

        // 逆変換は conj(DFT(conj(x))) の恒等式で順方向と同じ核を再利用する。
        for (int k = 0; k < n; k++)
        {
            LaneLoops.Apply(
                ref Unsafe.Add(ref input, batch * k), ref Unsafe.Add(ref u, batch * k), batch,
                new ChirpIn<TD>(chirp![k]));
        }

        MemoryMarshal.CreateSpan(ref Unsafe.Add(ref u, batch * n), (int)(batch * (m - n))).Clear();

        bool inV = inner.RunStages<ForwardDirection>(ref u, ref v, ref u, batch);
        ref Complex spectrum = ref inV ? ref v : ref u;
        ref Complex spare = ref inV ? ref u : ref v;

        for (int k = 0; k < m; k++)
        {
            ref Complex column = ref Unsafe.Add(ref spectrum, batch * k);
            LaneLoops.Apply(ref column, ref column, batch, new MultiplyBy(kernelSpectrum![k]));
        }

        bool inSpare = inner.RunStages<InverseDirection>(ref spectrum, ref spare, ref spectrum, batch);
        ref Complex convolution = ref inSpare ? ref spare : ref spectrum;

        for (int k = 0; k < n; k++)
        {
            LaneLoops.Apply(
                ref Unsafe.Add(ref convolution, batch * k), ref Unsafe.Add(ref output, batch * k), batch,
                new ChirpOut<TD>(chirp![k]));
        }
    }

    private readonly struct ChirpIn<TD>(Complex w) : ILaneOperation
        where TD : IFourierDirection
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Invoke<TL>(ref Complex first, ref Complex second) where TL : unmanaged, IComplexLanes<TL>
        {
            TL x = TL.Load(in first);
            if (TD.IsInverse)
            {
                x = TL.Conjugate(x);
            }

            TL.Store(ref second, TL.Multiply(x, w));
        }
    }

    private readonly struct ChirpOut<TD>(Complex w) : ILaneOperation
        where TD : IFourierDirection
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Invoke<TL>(ref Complex first, ref Complex second) where TL : unmanaged, IComplexLanes<TL>
        {
            TL x = TL.Multiply(TL.Load(in first), w);
            if (TD.IsInverse)
            {
                x = TL.Conjugate(x);
            }

            TL.Store(ref second, x);
        }
    }

    private readonly struct MultiplyBy(Complex w) : ILaneOperation
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Invoke<TL>(ref Complex first, ref Complex second) where TL : unmanaged, IComplexLanes<TL> =>
            TL.Store(ref second, TL.Multiply(TL.Load(in first), w));
    }
}

internal static class FourierPlanCache
{
    private const int Capacity = 64;
    private static readonly ConcurrentDictionary<int, FourierPlan> ComplexPlans = new();
    private static readonly ConcurrentDictionary<int, RealFourierPlan> RealPlans = new();

    internal static FourierPlan GetComplex(int length, Func<int, FourierPlan> factory) => Get(ComplexPlans, length, factory);

    internal static RealFourierPlan GetReal(int length, Func<int, RealFourierPlan> factory) => Get(RealPlans, length, factory);

    private static TPlan Get<TPlan>(ConcurrentDictionary<int, TPlan> cache, int length, Func<int, TPlan> factory)
    {
        if (cache.TryGetValue(length, out var plan))
        {
            return plan;
        }

        if (cache.Count >= Capacity)
        {
            cache.Clear();
        }

        return cache.GetOrAdd(length, factory);
    }
}
