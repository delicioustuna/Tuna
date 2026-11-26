using System.Diagnostics;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;

namespace Tuna.LinearAlgebra;

// 短い並列区間を高頻度に繰り返すための専用フォーク・ジョイン実行基盤（スピン待機後に休止するワーカープール）。
// 並列区間は同時に 1 つのみ許可し、入れ子や他スレッドからの同時呼び出しは逐次実行にフォールバックする。
internal static class Parallelism
{
    // 並列区間終了後のスピン待機時間（マイクロ秒）。
    private const double SpinMicroseconds = 2000;

    private static readonly int Workers = Environment.ProcessorCount - 1;

    private static WorkerPool? pool;

    private static int busy;

    // 1 スレッドあたりの最小演算量目安（積和回数、約 3us 相当）。この 2 倍未満は並列化しない。
    internal const double Grain = 128 * 32 * 32;

    [ThreadStatic]
    private static int sequentialDepth;

    [ThreadStatic]
    private static bool inWork;

    // 呼び出し元スレッドは 0、ワーカー w は w + 1。
    [ThreadStatic]
    private static int slot;

    // このスレッドがワーカープールによる並列区間の作業を実行中かどうか（入れ子の逐次実行も含む）。
    internal static bool InWork => inWork;

    // 現在の並列区間のスロット番号（InWork が true のときのみ有効）。
    internal static int Slot => slot;

    // スロット総数（呼び出し元 + ワーカー数）。
    internal static int SlotCount => Workers + 1;

    // 利用可能な最大スレッド数（Sequential スコープ内では 1）。
    internal static int MaxThreads => sequentialDepth > 0 ? 1 : Math.Min(LinearAlgebraSettings.MaxDegreeOfParallelism, Workers + 1);

    // 破棄されるまでこのスレッドからの並列化を抑止する（小規模部分問題での同期オーバーヘッド回避用）。
    internal static SequentialScope Sequential(bool enabled = true)
    {
        if (enabled)
        {
            sequentialDepth++;
        }

        return new SequentialScope(enabled);
    }

    internal readonly ref struct SequentialScope(bool enabled)
    {
        public void Dispose()
        {
            if (enabled)
            {
                sequentialDepth--;
            }
        }
    }

    // 演算量 work を grain 以上ずつに分割した並列度（1 なら逐次実行）。
    internal static int Degree(double work, double grain)
    {
        int max = MaxThreads;
        if (max <= 1 || work < 2 * grain || Volatile.Read(ref busy) != 0)
        {
            return 1;
        }

        return (int)Math.Min(max, work / grain);
    }

    // 作業配列プールのクリーンアップ用。占有中は他の並列区間を呼び出し元スレッドで逐次実行させる。
    internal static bool TryBeginExclusive() => Interlocked.CompareExchange(ref busy, 1, 0) == 0;

    internal static void EndExclusive() => Volatile.Write(ref busy, 0);

    // 他の並列区間実行中でも並列度を下げない（分割数依存の丸め誤差を実行時状態から独立させるため）。
    internal static int PlannedDegree(double work, double grain)
    {
        int max = MaxThreads;
        if (max <= 1 || work < 2 * grain)
        {
            return 1;
        }

        return (int)Math.Min(max, work / grain);
    }

    // 動的スケジュールで 0..count-1 を最大 threads スレッドで実行する。
    internal static void For(int count, int threads, Action<int> body)
    {
        threads = Math.Min(Math.Min(threads, count), MaxThreads);
        if (threads <= 1 || Interlocked.CompareExchange(ref busy, 1, 0) != 0)
        {
            for (int i = 0; i < count; i++)
            {
                body(i);
            }

            return;
        }

        try
        {
            pool ??= new WorkerPool(Workers);
            pool.Run(count, threads, body);
            WorkArrayPools.AfterRegion(threads);
        }
        finally
        {
            Volatile.Write(ref busy, 0);
        }
    }

    // 動的スケジュール（値型ワークによりクロージャ確保やボックス化を避ける）。
    internal static unsafe void For<TWork>(int count, int threads, ref TWork work)
        where TWork : unmanaged, IParallelWork
    {
        threads = Math.Min(Math.Min(threads, count), MaxThreads);
        if (threads <= 1 || Interlocked.CompareExchange(ref busy, 1, 0) != 0)
        {
            for (int i = 0; i < count; i++)
            {
                work.Invoke(i);
            }

            return;
        }

        try
        {
            pool ??= new WorkerPool(Workers);
            var context = WorkContext<TWork>.Instance;
            // 呼び出し元の状態を全作業の終了まで固定する。状態が持つ領域も、呼び出し元がその間固定しておく。
            fixed (TWork* state = &work)
            {
                Debug.Assert(context.State == null);
                context.State = state;
                try
                {
                    pool.Run(count, threads, context.Body);
                }
                finally
                {
                    context.State = null;
                }
            }

            WorkArrayPools.AfterRegion(threads);
        }
        finally
        {
            Volatile.Write(ref busy, 0);
        }
    }

    // ワーク型ごとにデリゲートを使い回す。busy による単一並列区間制約が前提（同時実行を許すと state ポインタが競合する）。
    private sealed unsafe class WorkContext<TWork>
        where TWork : unmanaged, IParallelWork
    {
        internal static readonly WorkContext<TWork> Instance = new();
        internal readonly Action<int> Body;
        internal TWork* State;

        private WorkContext()
        {
            Body = Invoke;
        }

        private void Invoke(int index) => State->Invoke(index);
    }

    // 静的スケジュール（t を常に同じスレッドへ割り当て L2 キャッシュ局所性を保つ。並列化不能時は呼び出し元で順次実行）。
    internal static void ForStatic(int threads, Action<int> body)
    {
        if (threads <= 1 || threads > MaxThreads || Interlocked.CompareExchange(ref busy, 1, 0) != 0)
        {
            for (int i = 0; i < threads; i++)
            {
                body(i);
            }

            return;
        }

        try
        {
            pool ??= new WorkerPool(Workers);
            pool.Run(threads, threads, body, staticSchedule: true);
            WorkArrayPools.AfterRegion(threads);
        }
        finally
        {
            Volatile.Write(ref busy, 0);
        }
    }

    // 値型ワーク向けの静的スケジュール。
    internal static unsafe void ForStatic<TWork>(int threads, ref TWork work)
        where TWork : unmanaged, IParallelWork
    {
        if (threads <= 1 || threads > MaxThreads || Interlocked.CompareExchange(ref busy, 1, 0) != 0)
        {
            for (int i = 0; i < threads; i++)
            {
                work.Invoke(i);
            }

            return;
        }

        try
        {
            pool ??= new WorkerPool(Workers);
            var context = WorkContext<TWork>.Instance;
            // 呼び出し元の状態を全作業の終了まで固定する。状態が持つ領域も、呼び出し元がその間固定しておく。
            fixed (TWork* state = &work)
            {
                Debug.Assert(context.State == null);
                context.State = state;
                try
                {
                    pool.Run(threads, threads, context.Body, staticSchedule: true);
                }
                finally
                {
                    context.State = null;
                }
            }

            WorkArrayPools.AfterRegion(threads);
        }
        finally
        {
            Volatile.Write(ref busy, 0);
        }
    }

    private sealed class WorkerPool
    {
        private static readonly long SpinTicks = (long)(SpinMicroseconds * Stopwatch.Frequency / 1e6);

        // 直前の並列区間に参加しなかったワーカーの短いスピン時間（SMT 競合を抑えるため）。
        private static readonly long IdleSpinTicks = SpinTicks / 8;

        private readonly SemaphoreSlim[] wake;

        private readonly int[] parked; // 休止中または休止遷移中なら 1

        private volatile int generation;
        private Action<int>? body;
        private int count;
        private bool staticSchedule;

        private readonly int[] assigned; // 各ワーカーが参加する並列区間の世代
        private WorkCounters counters;
        private Exception? error;

        // 偽共有を避けるため、頻繁に更新する next / remaining を別キャッシュラインに分離する。
        [StructLayout(LayoutKind.Explicit, Size = 512)]
        private struct WorkCounters
        {
            [FieldOffset(128)]
            internal int Next;

            [FieldOffset(256)]
            internal int Pending;
        }

        internal WorkerPool(int workers)
        {
            wake = new SemaphoreSlim[workers];
            parked = new int[workers];
            assigned = new int[workers];
            for (int w = 0; w < workers; w++)
            {
                wake[w] = new SemaphoreSlim(0);
                int id = w;
                var thread = new Thread(() => Loop(id))
                {
                    IsBackground = true,
                    Name = $"Tuna.LinearAlgebra worker {w}",
                };
                thread.Start();
            }
        }

        internal void Run(int count, int threads, Action<int> body, bool staticSchedule = false)
        {
            this.body = body;
            this.count = count;
            this.staticSchedule = staticSchedule;
            counters.Next = 0;
            error = null;
            counters.Pending = threads - 1;
            int g = generation + 1;
            for (int w = 0; w < threads - 1; w++)
            {
                Volatile.Write(ref assigned[w], g);
            }

            generation = g;
            for (int w = 0; w < threads - 1; w++)
            {
                if (Interlocked.Exchange(ref parked[w], 0) == 1)
                {
                    wake[w].Release();
                }
            }

            Execute(-1);
            var spin = default(SpinWait);
            ThreadInterruptedException? interruption = null;
            while (Volatile.Read(ref counters.Pending) > 0)
            {
                try
                {
                    spin.SpinOnce(-1);
                }
                catch (ThreadInterruptedException ex)
                {
                    // 作業が終わる前に戻ると、固定した領域や呼び出し元のスタックを作業スレッドが使い続けてしまう。
                    interruption ??= ex;
                }
            }

            this.body = null;
            if (interruption is not null)
            {
                ExceptionDispatchInfo.Throw(interruption);
            }

            if (error is not null)
            {
                throw new AggregateException(error);
            }
        }

        // id はワーカー番号（呼び出し元スレッドは -1）。
        private void Execute(int id)
        {
            inWork = true;
            slot = id + 1;
            try
            {
                if (staticSchedule)
                {
                    if (id + 1 < count)
                    {
                        body!(id + 1);
                    }

                    return;
                }

                int i;
                while ((i = Interlocked.Increment(ref counters.Next) - 1) < count)
                {
                    body!(i);
                }
            }
            catch (Exception ex)
            {
                Interlocked.CompareExchange(ref error, ex, null);
                Volatile.Write(ref counters.Next, count);
            }
            finally
            {
                inWork = false;
            }
        }

        private void Loop(int id)
        {
            int seen = 0;
            bool participated = false;
            while (true)
            {
                long start = Stopwatch.GetTimestamp();
                long spinTicks = participated ? SpinTicks : IdleSpinTicks;
                int spins = 0;
                while (generation == seen)
                {
                    // スピン中も定期的に CPU を譲る。スレッド数が使えるコアを超えたとき（SMT の相方や他の負荷）、
                    // 譲らないと割り込まれた作業スレッドが CPU に戻れず、静的に分担した区間の全員がその復帰を待つ
                    if (Stopwatch.GetTimestamp() - start < spinTicks)
                    {
                        if ((++spins & 15) == 0)
                        {
                            Thread.Yield();
                        }
                        else
                        {
                            Thread.SpinWait(20);
                        }

                        continue;
                    }

                    // 休止する。休止を宣言した後に世代が進んでいたら、起こす側と競合しないように自分で取り消す
                    // 宣言の store と世代の load の間には完全なフェンスが必要。
                    // Volatile.Write (release) だけでは store→load の並べ替えを防げず、
                    // 起こす側が parked=0、こちらが古い generation を同時に読んで
                    // Release を逃したまま Wait に入ることがある。
                    Interlocked.Exchange(ref parked[id], 1);
                    if (generation != seen && Interlocked.Exchange(ref parked[id], 0) == 1)
                    {
                        break;
                    }

                    wake[id].Wait();
                    start = Stopwatch.GetTimestamp();
                }

                seen = generation;
                participated = Volatile.Read(ref assigned[id]) == seen;
                if (participated)
                {
                    Execute(id);
                    Interlocked.Decrement(ref counters.Pending);
                }
            }
        }
    }
}

// 並列区間で実行する値型ワーク。
internal interface IParallelWork
{
    void Invoke(int index);
}

// 固定済みメモリ領域を並列スレッドへ渡すポインタと長さ（Span<T> はキャプチャ不可のため）。
internal readonly unsafe struct SharedSpan
{
    private readonly double* pointer;
    private readonly int length;

    internal SharedSpan(double* pointer, int length)
    {
        this.pointer = pointer;
        this.length = length;
    }

    internal Span<double> Span => new(pointer, length);
}
