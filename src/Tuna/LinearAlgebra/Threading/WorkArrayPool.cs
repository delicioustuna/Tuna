using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Tuna.LinearAlgebra;

// 共有の ArrayPool<T>.Shared は、返した配列をまずそのスレッドの手元に置き、共有の置き場はスレッドの間で取り合う。
// 並列区間の作業の中（入れ子で逐次に実行する行列積なども含む）で借りると、どのスレッドがどの作業を受け持つかが回ごとに変わるので、
// 同じ計算を繰り返しても、呼び出し元のスレッドが借りるときに手元にも共有の置き場にも配列がなく、まれに確保していた。
// ここでは作業の中で借りる配列を、作業を実行しているスロット（呼び出し元は 0、作業スレッド w は w + 1）ごとの置き場から出す。
// 置き場はそのスロットのスレッドだけが区間の中で触り、足りなかった大きさは区間の後に呼び出し元が、それまでに区間に参加したすべてのスロットに足す
// （それまでより多くのスロットが参加した区間の後も、増えたスロットに足す）。
// 足す数は、1つの作業の中で同時に借りた数の最大なので、どのスロットがどの作業を受け持っても足り、同じ計算を繰り返すと2回目以降は作業の中で確保しない。
// ただし区間の後に足す（まだ使っていないスロットのために先に確保する）のは、すべての型・スロットを合わせて WorkArrayPools.FillLimitBytes までにし、
// 小さい区分から、区分ごとにスロットの順に足す。足さなかった配列は、そのスロットの作業が実際に借りるときに共有のプールから借りる
// （返した配列はそのスロットの置き場に入れ、上限には数えない）。
// 先に確保する配列は計算に要らないので、確保できなければ（GC ヒープの上限など）足すのをやめ、先に確保した配列を手放して計算の結果を返す。
// 作業の外では ArrayPool<T>.Shared と同じ。大きさの区分（16 から 2 倍ずつ 2^30 まで）も同じにする。
/// <summary>
/// 並列区間の作業の中で使う配列のプール
/// </summary>
internal static class WorkArrayPool<T>
{
    // 作業の中で初めて借りたときに作る（作業の外だけで使う型では作らない）
    private static readonly Slot[] Slots = Create();

    // 作業の中では、そのスロットの置き場から出す。
    /// <summary>
    /// 長さ <paramref name="minimumLength"/> 以上の配列を借りる（<see cref="ArrayPool{T}.Rent(int)"/> と同じ）
    /// </summary>
    internal static T[] Rent(int minimumLength)
    {
        if (!Parallelism.InWork)
        {
            return ArrayPool<T>.Shared.Rent(minimumLength);
        }

        int bucket = WorkArrayPools.SelectBucket(minimumLength);
        if ((uint)bucket >= WorkArrayPools.Buckets)
        {
            return ArrayPool<T>.Shared.Rent(minimumLength);
        }

        return Slots[Parallelism.Slot].Rent(bucket);
    }

    // 借りたときと同じく作業の中か外かで返す先を選ぶので、作業の中で借りたものは同じ作業の中で返す。
    /// <summary>
    /// <see cref="Rent"/> で借りた配列を返す（<see cref="ArrayPool{T}.Return(T[], bool)"/> と同じ）
    /// </summary>
    internal static void Return(T[] array, bool clearArray = false)
    {
        int bucket = WorkArrayPools.SelectBucket(array.Length);
        if (!Parallelism.InWork || (uint)bucket >= WorkArrayPools.Buckets || array.Length != WorkArrayPools.BucketLength(bucket))
        {
            ArrayPool<T>.Shared.Return(array, clearArray);
            return;
        }

        if (clearArray)
        {
            Array.Clear(array);
        }

        Slots[Parallelism.Slot].Return(bucket, array);
    }

    private static Slot[] Create()
    {
        var slots = new Slot[Parallelism.SlotCount];
        for (int s = 0; s < slots.Length; s++)
        {
            slots[s] = new Slot();
        }

        WorkArrayPools.Register(new Pool(slots));
        return slots;
    }

    // 区間の後と片付けのときに WorkArrayPools から呼ぶ（どちらも並列区間を占有している）
    /// <summary>
    /// 型ごとの置き場をまとめて扱う
    /// </summary>
    private sealed class Pool(Slot[] slots) : WorkArrayPools.IPool
    {
        // 小さい区分から、区分ごとにスロットの順に足す（上限に当たっても、小さい配列はなるべく多くのスロットに足りるように）
        public void TopUp(int threads, ref long budget)
        {
            int count = Math.Min(threads, slots.Length);
            for (int bucket = 0; bucket < WorkArrayPools.Buckets; bucket++)
            {
                int need = 0;
                foreach (var slot in slots)
                {
                    need = Math.Max(need, slot.Need[bucket]);
                }

                for (int s = 0; s < count && need > 0; s++)
                {
                    slots[s].Fill(bucket, need, ref budget);
                }
            }
        }

        public long FilledBytes()
        {
            long bytes = 0;
            foreach (var slot in slots)
            {
                bytes += slot.FilledBytes();
            }

            return bytes;
        }

        // 作業が例外で終わって返さなかった分は数えない
        public void ResetHeld()
        {
            foreach (var slot in slots)
            {
                slot.ResetHeld();
            }
        }

        public void ReleaseFilled()
        {
            foreach (var slot in slots)
            {
                slot.ReleaseFilled();
            }
        }

        // 区分ごとに、どのスロットでも使っていなければ、すべてのスロットで手放す（一部のスロットだけ手放すと、次に足すときに戻す）
        public void Trim(long now, bool all)
        {
            for (int bucket = 0; bucket < WorkArrayPools.Buckets; bucket++)
            {
                long used = long.MinValue;
                foreach (var slot in slots)
                {
                    used = Math.Max(used, slot.Used(bucket));
                }

                if (all || now - used >= WorkArrayPools.TrimAfterMilliseconds)
                {
                    foreach (var slot in slots)
                    {
                        slot.Release(bucket);
                    }
                }
            }
        }
    }

    /// <summary>
    /// 1つのスロットの置き場（大きさの区分ごとの積み重ね）
    /// </summary>
    private sealed class Slot
    {
        private readonly T[]?[]?[] stacks = new T[]?[]?[WorkArrayPools.Buckets];
        private readonly int[] counts = new int[WorkArrayPools.Buckets];

        /// <summary>
        /// 借りていてまだ返していない数
        /// </summary>
        private readonly int[] held = new int[WorkArrayPools.Buckets];

        /// <summary>
        /// 最後に借りた時刻（<see cref="Environment.TickCount64"/>）
        /// </summary>
        private readonly long[] used = new long[WorkArrayPools.Buckets];

        /// <summary>
        /// 区間の後に足した（先に確保した）数（手放すまで減らさない）
        /// </summary>
        private readonly int[] filled = new int[WorkArrayPools.Buckets];

        /// <summary>
        /// 1つの作業の中で同時に借りた数の最大（置き場が足りなかったときに更新する）
        /// </summary>
        internal readonly int[] Need = new int[WorkArrayPools.Buckets];

        // 足りなければ共有のプールから借り、返すときにこの置き場に入れる。
        internal T[] Rent(int bucket)
        {
            used[bucket] = Environment.TickCount64;
            int count = counts[bucket];
            int inUse = ++held[bucket];
            if (count > 0)
            {
                var stack = stacks[bucket]!;
                var array = stack[--count]!;
                stack[count] = null;
                counts[bucket] = count;
                return array;
            }

            if (inUse > Need[bucket])
            {
                Need[bucket] = inUse;
            }

            WorkArrayPools.ReportShortage();
            return ArrayPool<T>.Shared.Rent(WorkArrayPools.BucketLength(bucket));
        }

        // 積み重ねを広げる（確保する）のは、足りなかった区間の中か区間の後だけ。
        internal void Return(int bucket, T[] array)
        {
            held[bucket]--;
            Push(bucket, array);
        }

        // 区間の後に呼び出し元が呼ぶ。返していない配列はないので、置き場の数が持っている数になる。
        /// <summary>
        /// 区分 <paramref name="bucket"/> の置き場の数を、足した配列の大きさの合計が <paramref name="budget"/> を超えない分だけ <paramref name="need"/> まで足す
        /// </summary>
        internal void Fill(int bucket, int need, ref long budget)
        {
            long bytes = (long)WorkArrayPools.BucketLength(bucket) * Unsafe.SizeOf<T>();
            while (counts[bucket] < need && bytes <= budget)
            {
                used[bucket] = Environment.TickCount64;
                Push(bucket, GC.AllocateUninitializedArray<T>(WorkArrayPools.BucketLength(bucket)));
                filled[bucket]++;
                budget -= bytes;
            }
        }

        // 作業が例外で終わって返さなかった配列は、置き場の数から除く
        /// <summary>
        /// 区間の後に足した配列のうち、置き場にある分の大きさの合計（バイト）
        /// </summary>
        internal long FilledBytes()
        {
            long bytes = 0;
            for (int bucket = 0; bucket < filled.Length; bucket++)
            {
                bytes += (long)Math.Min(filled[bucket], counts[bucket]) * WorkArrayPools.BucketLength(bucket);
            }

            return bytes * Unsafe.SizeOf<T>();
        }

        internal void ResetHeld() => Array.Clear(held);

        // 区分ごとに、区間の後に足した数だけ置き場の配列を手放す（同じ区分の配列はどれも同じ長さなので、どれを手放してもよい）
        internal void ReleaseFilled()
        {
            for (int bucket = 0; bucket < filled.Length; bucket++)
            {
                int count = counts[bucket], release = Math.Min(filled[bucket], count);
                if (release > 0)
                {
                    Array.Clear(stacks[bucket]!, count - release, release);
                    counts[bucket] = count - release;
                }

                filled[bucket] = 0;
            }
        }

        internal long Used(int bucket) => used[bucket];

        // 足す数も忘れる（手放した区分を、ほかの区分が足りなかったときに足し直さない。また借りて足りなければ覚え直す）
        /// <summary>
        /// 区分 <paramref name="bucket"/> の配列を手放す
        /// </summary>
        internal void Release(int bucket)
        {
            if (counts[bucket] > 0)
            {
                Array.Clear(stacks[bucket]!, 0, counts[bucket]);
                counts[bucket] = 0;
            }

            filled[bucket] = 0;
            Need[bucket] = 0;
        }

        private void Push(int bucket, T[] array)
        {
            int count = counts[bucket];
            ref var stack = ref stacks[bucket];
            if (stack is null || count == stack.Length)
            {
                Array.Resize(ref stack, Math.Max(4, 2 * count));
            }

            stack[count] = array;
            counts[bucket] = count + 1;
        }
    }
}

/// <summary>
/// <see cref="WorkArrayPool{T}"/> の型によらない部分（足りなかった区分を足すことと片付け）
/// </summary>
internal static class WorkArrayPools
{
    /// <summary>
    /// 大きさの区分の数（16 から 2 倍ずつ 2^30 まで。<see cref="ArrayPool{T}.Shared"/> と同じ）
    /// </summary>
    internal const int Buckets = 27;

    // ArrayPool<T>.Shared の目安と同じ。
    /// <summary>
    /// この時間使わなかった区分の配列は、gen2 の GC の後に手放す
    /// </summary>
    internal const long TrimAfterMilliseconds = 60_000;

    // スロットの数（並列数）によらない上限にする（1スロットに型ごとに 4 MiB の上限では、並列数 32 で 62 MiB を先に確保した。#136 のレビュー）。
    // 値は、並列数 4 でその上限（4 スロット分）と同じ量にする。先に確保した配列は使わなくなっても片付けるまで上限に数えるので、
    // 上限が小さいと、前に使った形の配列で埋まって次の形の配列を足せない（4 MiB では、AD のテストの順で上限がほぼ埋まり、
    // 全体のテストの1回で 300 次の線形代数のモデルがくり返しの中で確保した）。
    // 上限を超える分は、作業が実際に借りるときに共有のプールから借りる（確保しても計算に比べて小さい）。
    /// <summary>
    /// 区間の後に先に確保して足した配列の大きさの合計の上限（すべての型・スロットを合わせる、バイト）
    /// </summary>
    internal const long FillLimitBytes = 16L << 20;

    private static readonly List<IPool> Pools = [];

    /// <summary>
    /// 作業の中で置き場が足りなかったら 1
    /// </summary>
    private static int shortage;

    /// <summary>
    /// 作業の中で置き場が足りなかった回数
    /// </summary>
    private static long shortages;

    /// <summary>
    /// 片付けを次の区間の後に回したら 1（2 ならメモリが逼迫していて、すべて手放す）
    /// </summary>
    private static int trimRequest;

    /// <summary>
    /// 最後に並列区間を終えた時刻（<see cref="Environment.TickCount64"/>）
    /// </summary>
    private static long lastRegion;

    /// <summary>
    /// 足りなかった区分を足すスロットの数（それまでの区間に参加したスレッド数の最大。区間の後に呼び出し元だけが触る）
    /// </summary>
    private static int filledThreads;

    /// <summary>
    /// この時刻（<see cref="Environment.TickCount64"/>）までは先に確保しない（確保できなかった時刻から <see cref="TrimAfterMilliseconds"/> 後。区間の後に呼び出し元だけが触る）
    /// </summary>
    private static long fillSuspendedUntil = long.MinValue;

    /// <summary>
    /// 型ごとの置き場
    /// </summary>
    internal interface IPool
    {
        /// <summary>
        /// スロット <c>[0, threads)</c> に、足りなかった区分の配列を、足した大きさの合計が <paramref name="budget"/> を超えない分だけ足す（足した分を引く）
        /// </summary>
        void TopUp(int threads, ref long budget);

        /// <summary>
        /// 区間の後に足した配列のうち、置き場にある分の大きさの合計（バイト）
        /// </summary>
        long FilledBytes();

        /// <summary>
        /// 借りていてまだ返していない数を 0 に戻す
        /// </summary>
        void ResetHeld();

        /// <summary>
        /// 区間の後に先に確保して足した配列を手放す
        /// </summary>
        void ReleaseFilled();

        /// <summary>
        /// 使わない区分（<paramref name="all"/> なら全部）の配列を手放す
        /// </summary>
        void Trim(long now, bool all);
    }

    internal static int SelectBucket(int length) => BitOperations.Log2((uint)length - 1 | 15) - 3;

    internal static int BucketLength(int bucket) => 16 << bucket;

    // テストが、同じ計算をくり返したときに置き場が足りなくならないことを数えるのに使う。
    /// <summary>
    /// 作業の中で置き場が足りなかった回数
    /// </summary>
    internal static long Shortages => Interlocked.Read(ref shortages);

    internal static void ReportShortage()
    {
        Interlocked.Increment(ref shortages);
        Volatile.Write(ref shortage, 1);
    }

    internal static void Register(IPool pool)
    {
        lock (Pools)
        {
            if (Pools.Count == 0)
            {
                _ = new Gen2Callback();
            }

            Pools.Add(pool);
        }
    }

    // Parallelism が区間の作業がすべて終わった後、区間を占有したまま呼び出し元のスレッドで呼ぶ。
    // 参加したスロットが増えた区間の後にも足す。足りなかった区分を足した区間より多くのスロットが参加した区間で、
    // たまたま足りていた（増えたスロットがその区分を借りる作業を受け持たなかった）場合に、次の回に足りなくならないように。
    // 先に確保できなかったら、TrimAfterMilliseconds の間は先に確保しない（確保できない間、区間のたびに GC をくり返さないように）。
    /// <summary>
    /// 区間の中で足りなかった区分を、それまでの区間に参加したすべてのスロットに足す（この区間には <paramref name="threads"/> 個のスロットが参加した）
    /// </summary>
    internal static void AfterRegion(int threads)
    {
        bool wider = threads > filledThreads;
        if (wider)
        {
            filledThreads = threads;
        }

        long now = Environment.TickCount64;
        if (wider || Volatile.Read(ref shortage) != 0)
        {
            Volatile.Write(ref shortage, 0);
            lock (Pools)
            {
                if (now >= fillSuspendedUntil && !TryTopUp())
                {
                    fillSuspendedUntil = now + TrimAfterMilliseconds;
                }

                foreach (var pool in Pools)
                {
                    pool.ResetHeld();
                }
            }
        }

        int request = Interlocked.Exchange(ref trimRequest, 0);
        if (request != 0)
        {
            TrimAll(now, request == 2);
        }

        Volatile.Write(ref lastRegion, now);
    }

    // 区間の作業はもう終わっていて、先に確保する配列は計算に要らない。確保できなければ（GC ヒープの上限など）足すのをやめ、
    // 計算の結果を返せるように、先に確保した配列を手放して偽を返す（呼び出し元は、しばらく先に確保しない）。
    // 上限の中でも、ヒープの残りが少なければ確保できないことがある（#136 のレビュー）。
    private static bool TryTopUp()
    {
        long budget = FillLimitBytes;
        foreach (var pool in Pools)
        {
            budget -= pool.FilledBytes();
        }

        try
        {
            foreach (var pool in Pools)
            {
                pool.TopUp(filledThreads, ref budget);
            }

            return true;
        }
        catch (OutOfMemoryException)
        {
            foreach (var pool in Pools)
            {
                pool.ReleaseFilled();
            }

            return false;
        }
    }

    private static void TrimAll(long now, bool all)
    {
        lock (Pools)
        {
            foreach (var pool in Pools)
            {
                pool.Trim(now, all);
            }
        }
    }

    // ArrayPool<T>.Shared と同じく、物理メモリの使用量が逼迫の閾値の 90% を超えたら高い。
    private static bool HighMemoryPressure()
    {
        var info = GC.GetGCMemoryInfo();
        return info.MemoryLoadBytes >= info.HighMemoryLoadThresholdBytes * 0.9;
    }

    // 並列区間の実行中に片付けると、そのスロットのスレッドと同時に置き場を触るので、区間の後に回す。
    // 並列区間をしばらく使っていないとき（またはメモリが逼迫しているとき）は、占有できればここで片付ける。
    /// <summary>
    /// gen2 の GC のたびに終了処理が呼ばれ、置き場を片付ける
    /// </summary>
    private sealed class Gen2Callback
    {
        ~Gen2Callback()
        {
            long now = Environment.TickCount64;
            bool high = HighMemoryPressure();
            if ((high || now - Volatile.Read(ref lastRegion) >= TrimAfterMilliseconds) && Parallelism.TryBeginExclusive())
            {
                try
                {
                    TrimAll(now, high);
                }
                finally
                {
                    Parallelism.EndExclusive();
                }
            }
            else
            {
                Volatile.Write(ref trimRequest, high ? 2 : 1);
            }

            GC.ReRegisterForFinalize(this);
        }
    }
}
