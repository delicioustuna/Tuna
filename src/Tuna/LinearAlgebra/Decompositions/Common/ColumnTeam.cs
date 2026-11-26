using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Tuna.LinearAlgebra;

// 列範囲ごとに担当スレッドを固定（owner-computes）し、縮約の行列ベクトル積と更新で L2 キャッシュ局所性を保つ。
// 部分和を使う場合の区間数・境界は次数のみで決定し（スレッド数非依存）、Rent/Return でスレッドごとに再利用する。
internal sealed unsafe class ColumnTeam
{
    // 使い回す1つ。境界と部分和の領域は、そのスレッドで使ったうち最大の大きさになる（部分和は行数の 1/256 程度の行列 1 枚以下）
    [ThreadStatic]
    private static ColumnTeam? cached;

    private readonly int[] bounds;
    private readonly int* boundsPointer;
    private readonly double[] partial;
    private readonly double* partialPointer;
    private readonly Action<int> transposedBody;
    private readonly Action<int> multiplyBody;
    private readonly Action<int> symmetricBody;
    private readonly Action<int> partitionBody;
    private Action<int>? body;
    private int maxRows;

    // 現在の呼び出しの引数
    private double* a;
    private int lda;
    private int rows;
    private int c0;
    private int c1;
    private double* x;
    private double* y;

    // maxRows == 0 なら部分和領域を確保せず列分割のみ行う（QR 更新用）。
    internal ColumnTeam(int threads, int maxRows)
        : this(threads, maxRows, 0, 0)
    {
    }

    private ColumnTeam(int threads, int maxRows, int boundsCapacity, int partialCapacity)
    {
        int parts = Count(threads, maxRows);
        bounds = GC.AllocateArray<int>(Math.Max(parts + 1, boundsCapacity), pinned: true);
        boundsPointer = (int*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(bounds));
        partial = GC.AllocateUninitializedArray<double>(Math.Max(checked((parts - 1) * maxRows), partialCapacity), pinned: true);
        partialPointer = (double*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(partial));
        transposedBody = TransposedTask;
        multiplyBody = MultiplyTask;
        symmetricBody = SymmetricTask;
        partitionBody = PartitionTask;
        Reset(threads, maxRows);
    }

    // 部分和の個数と境界は次数だけで決める。連続する区間を同じスレッドに割り当て、列の局所性を保つ。
    private static int Count(int threads, int maxRows) => maxRows == 0 ? threads : PartCount(maxRows);

    private void Reset(int threads, int maxRows)
    {
        Parts = Count(threads, maxRows);
        Threads = Math.Min(threads, Parts);
        this.maxRows = maxRows;
    }

    // スレッドキャッシュから再利用して用意する（入れ子呼び出し時は新規確保）。
    internal static ColumnTeam Rent(int threads, int maxRows)
    {
        int parts = Count(threads, maxRows);
        var team = cached;
        cached = null;
        if (team is not null && team.bounds.Length >= parts + 1 && team.partial.Length >= (long)(parts - 1) * maxRows)
        {
            team.Reset(threads, maxRows);
            return team;
        }

        return new ColumnTeam(threads, maxRows, team?.bounds.Length ?? 0, team?.partial.Length ?? 0);
    }

    internal static void Return(ColumnTeam team) => cached = team;

    internal const int PartUnit = 12; // 2/3/4/6/12 並列で均等分割できる単位

    // 並列数非依存の列区間数（600 行未満は部分和コストを抑えるため 6、それ以上は 3072 行ごとに 12 ずつ増加）。
    internal static int PartCount(int maxRows) => maxRows < 600 ? PartUnit / 2 : PartUnit * (1 + (maxRows - 1) / 3072);

    internal int Threads { get; private set; }
    internal int Parts { get; private set; }
    internal ColumnRanges Ranges => new(boundsPointer);

    internal void For(Action<int> action)
    {
        body = action;
        try
        {
            Parallelism.ForStatic(Threads, partitionBody);
        }
        finally
        {
            body = null;
        }
    }

    internal void For<TWork>(ref TWork work)
        where TWork : unmanaged, IParallelWork
    {
        fixed (TWork* state = &work)
        {
            var partition = new PartitionWork<TWork>(state, Parts, Threads);
            Parallelism.ForStatic(Threads, ref partition);
        }
    }

    private readonly struct PartitionWork<TWork>(TWork* work, int parts, int threads) : IParallelWork
        where TWork : unmanaged, IParallelWork
    {
        public void Invoke(int t)
        {
            int end = (int)((long)(t + 1) * parts / threads);
            for (int part = (int)((long)t * parts / threads); part < end; part++)
            {
                work->Invoke(part);
            }
        }
    }

    private void PartitionTask(int t)
    {
        int end = (int)((long)(t + 1) * Parts / Threads);
        for (int part = (int)((long)t * Parts / Threads); part < end; part++)
        {
            body!(part);
        }
    }

    // 列 [0, n) を 4 列単位で重み均等分割する（先頭 lead 列は区間 0 に含め、triangular 時は下三角長 n - c で重み付け）。
    internal void Partition(int n, int lead, bool triangular = false, double leadWeight = 0.5)
    {
        lead = Math.Min(lead, n);
        double total = 0;
        for (int c = 0; c < n; c++)
        {
            total += Weight(c, n, lead, triangular, leadWeight);
        }

        bounds[0] = 0;
        int t = 1;
        double sum = 0;
        for (int c = 0; c < n && t < Parts; c++)
        {
            sum += Weight(c, n, lead, triangular, leadWeight);
            while (t < Parts && sum >= total * t / Parts)
            {
                bounds[t] = Math.Clamp((c + 3) & ~3, Math.Max(bounds[t - 1], lead), n);
                t++;
            }
        }

        for (; t <= Parts; t++)
        {
            bounds[t] = n;
        }
    }

    private static double Weight(int c, int n, int lead, bool triangular, double leadWeight) =>
        (c < lead ? leadWeight : 1.0) * (triangular ? n - c : 1);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal (int Lo, int Hi) Range(int t, int from) => (Math.Max(bounds[t], from), bounds[t + 1]);

    // y[c - c0] = A[0:rows, c]^T x（c in [c0, c1)）。
    internal void MultiplyTransposed(double* a, int lda, int rows, int c0, int c1, double* x, double* y)
    {
        Set(a, lda, rows, c0, c1, x, y);
        For(transposedBody);
    }

    // y = Σ A[0:rows, c] x[c - c0]（各列区間の部分和を区間順に合算）。
    internal void Multiply(double* a, int lda, int rows, int c0, int c1, double* x, double* y)
    {
        Set(a, lda, rows, c0, c1, x, y);
        For(multiplyBody);
        AddPartials(y, rows, c0, c1, lower: false);
    }

    // y = A[c0:n, c0:n] x（下三角格納対称行列、部分和を区間順に合算）。
    internal void SymmetricMultiply(double* a, int lda, int n, int c0, double* x, double* y)
    {
        Set(a, lda, n, c0, n, x, y);
        For(symmetricBody);
        AddPartials(y, n - c0, c0, n, lower: true);
    }

    internal double* Partial(int t) => partialPointer + (nint)(t - 1) * maxRows;

    // 区間 1 以降の部分和を順に y へ加算する（[from, to) 外の区間はスキップ、lower 時は行 lo - from 開始）。
    internal void AddPartials(double* y, int rows, int from, int to, bool lower)
    {
        for (int t = 1; t < Parts; t++)
        {
            var (lo, hi) = Range(t, from);
            if (Math.Min(hi, to) <= lo)
            {
                continue;
            }

            int start = lower ? lo - from : 0;
            Level1Kernels.Axpy(1.0, new ReadOnlySpan<double>(Partial(t) + start, rows - start), new Span<double>(y + start, rows - start));
        }
    }

    private void SymmetricTask(int t)
    {
        int len = rows - c0;
        var (lo, hi) = Range(t, c0);
        if (hi <= lo)
        {
            if (t == 0)
            {
                new Span<double>(y, len).Clear();
            }

            return;
        }

        // 呼び出し元（0 番）は先頭から担当するので y に直接書く。他は作業ベクトルの担当列より下だけを書く
        var dest = new Span<double>(t == 0 ? y : Partial(t), len);
        double* origin = a + c0 + (nint)c0 * lda;
        MatrixVector.SymmetricMultiplySequential(
            lo - c0, hi - c0, len, 1.0, new ReadOnlySpan<double>(origin, (len - 1) * lda + len), lda, new ReadOnlySpan<double>(x, len), dest);
    }

    private void Set(double* a, int lda, int rows, int c0, int c1, double* x, double* y)
    {
        this.a = a;
        this.lda = lda;
        this.rows = rows;
        this.c0 = c0;
        this.c1 = c1;
        this.x = x;
        this.y = y;
    }

    private void TransposedTask(int t)
    {
        var (lo, hi) = Range(t, c0);
        hi = Math.Min(hi, c1);
        if (hi <= lo || rows <= 0)
        {
            return;
        }

        double* col = a + (nint)lo * lda;
        MatrixVector.MultiplyTransposedSequential(
            rows, hi - lo, new ReadOnlySpan<double>(col, (hi - lo - 1) * lda + rows), lda,
            new ReadOnlySpan<double>(x, rows), new Span<double>(y + (lo - c0), hi - lo));
    }

    private void MultiplyTask(int t)
    {
        var (lo, hi) = Range(t, c0);
        hi = Math.Min(hi, c1);
        if (hi <= lo)
        {
            // 空の区間の部分和は AddPartials が飛ばすので、y（0 番）だけを0にする
            if (t == 0)
            {
                new Span<double>(y, rows).Clear();
            }

            return;
        }

        var dest = new Span<double>(t == 0 ? y : Partial(t), rows);

        double* col = a + (nint)lo * lda;
        MatrixVector.MultiplySequential(
            rows, hi - lo, new ReadOnlySpan<double>(col, (hi - lo - 1) * lda + rows), lda,
            new ReadOnlySpan<double>(x + (lo - c0), hi - lo), dest);
    }
}

internal readonly unsafe struct ColumnRanges(int* bounds)
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal (int Lo, int Hi) Range(int t, int from) => (Math.Max(bounds[t], from), bounds[t + 1]);
}
