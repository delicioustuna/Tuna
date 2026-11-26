using System.Runtime.CompilerServices;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

internal sealed class DecompositionPool
{
    private const int RetentionFactor = 3;

    private readonly Dictionary<Key, Entry> entries = [];
    private Active[] active = [];
    private int count;
    private int previousCount;
    private Entry[] trimOrder = [];
    private long taken;
    private long peakTaken;
    private long retained;
    private long epoch;
    private bool created;

    public int Count => count;

    public int TakeCholesky(int n) => Take(DecompositionKind.Cholesky, n, n);

    public int TakeLu(int n) => Take(DecompositionKind.Lu, n, n);

    public int TakeQr(int m, int n) => Take(DecompositionKind.Qr, m, n);

    public int TakeSymmetricEigen(int n) => Take(DecompositionKind.SymmetricEigen, n, n);

    public int TakeSingularValue(int m, int n) => Take(DecompositionKind.SingularValue, m, n);

    public void ReleaseLast(int index)
    {
        ref Active item = ref active[index];
        if (index != count - 1)
        {
            throw new InvalidOperationException("Only the last decomposition can be released.");
        }

        item.Entry.Used--;
        taken -= item.Entry.Elements;
        item = default;
        count--;
    }

    public CholeskyDecomposition Cholesky(int index) => Unsafe.As<CholeskyDecomposition>(active[index].Decomposition);

    public LuDecomposition Lu(int index) => Unsafe.As<LuDecomposition>(active[index].Decomposition);

    public QrDecomposition Qr(int index) => Unsafe.As<QrDecomposition>(active[index].Decomposition);

    public SymmetricEigenDecomposition SymmetricEigen(int index) => Unsafe.As<SymmetricEigenDecomposition>(active[index].Decomposition);

    public SingularValueDecomposition SingularValue(int index) => Unsafe.As<SingularValueDecomposition>(active[index].Decomposition);

    public ref DecompositionInfo Info(int index) => ref active[index].Info;

    public void Reset()
    {
        if (created || count != previousCount)
        {
            Trim();
        }

        for (int i = 0; i < count; i++)
        {
            active[i].Entry.Used = 0;
            active[i] = default;
        }

        previousCount = count;
        count = 0;
        taken = 0;
        created = false;
        epoch++;
    }

    private int Take(DecompositionKind kind, int m, int n)
    {
        var key = new Key(kind, m, n);
        if (!entries.TryGetValue(key, out var entry))
        {
            entry = new Entry(Elements(kind, m, n));
            entries.Add(key, entry);
        }

        object decomposition;
        if (entry.Used < entry.Count)
        {
            decomposition = entry.Items[entry.Used];
        }
        else
        {
            decomposition = Create(kind, m, n);
            entry.Add(decomposition);
            retained += entry.Elements;
            created = true;
        }

        entry.Used++;
        entry.LastUse = epoch;
        taken += entry.Elements;
        // Track peak on acquisition so immediately released one-shot solves still count toward retention budget.
        peakTaken = Math.Max(peakTaken, taken);
        if (count == active.Length)
        {
            Array.Resize(ref active, Math.Max(4, 2 * active.Length));
        }

        active[count] = new Active { Decomposition = decomposition, Entry = entry };
        return count++;
    }

    private static long Elements(DecompositionKind kind, int m, int n)
    {
        long k = Math.Min(m, n);
        return kind switch
        {
            DecompositionKind.Qr => (long)m * n + m * k + k * n + k,
            DecompositionKind.SymmetricEigen => (long)n * n + n,
            DecompositionKind.SingularValue => k + m * k + n * k,
            _ => (long)n * n,
        };
    }

    // Seed each pooled decomposition with an owned backing matrix so subsequent calls reuse it in place via Recompute.
    private static object Create(DecompositionKind kind, int m, int n) => kind switch
    {
        DecompositionKind.Cholesky => CreateCholesky(n),
        DecompositionKind.Lu => CreateLu(n),
        DecompositionKind.Qr => QrDecomposition.Compute(new Matrix<double>(m, n), overwriteInput: true),
        DecompositionKind.SymmetricEigen => SymmetricEigenDecomposition.Compute(new Matrix<double>(n, n), computeVectors: true, overwriteInput: true),
        _ => SingularValueDecomposition.Compute(new Matrix<double>(m, n), computeVectors: true, overwriteInput: true),
    };

    private static CholeskyDecomposition CreateCholesky(int n)
    {
        var identity = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++)
        {
            identity[i, i] = 1;
        }

        CholeskyDecomposition.TryCompute(identity, identity, out var result);
        return result;
    }

    private static LuDecomposition CreateLu(int n) => LuDecomposition.Compute(new Matrix<double>(n, n), overwriteInput: true);

    private void Trim()
    {
        long limit = RetentionFactor * peakTaken;
        if (retained > limit)
        {
            if (trimOrder.Length < entries.Count)
            {
                trimOrder = new Entry[entries.Count];
            }

            int candidates = 0;
            foreach (var entry in entries.Values)
            {
                if (entry.Count > entry.Used)
                {
                    trimOrder[candidates++] = entry;
                }
            }

            Span<Entry> order = trimOrder.AsSpan(0, candidates);
            order.Sort(static (x, y) => x.LastUse.CompareTo(y.LastUse));
            foreach (var entry in order)
            {
                while (retained > limit && entry.Count > entry.Used)
                {
                    entry.RemoveLast();
                    retained -= entry.Elements;
                }

                if (retained <= limit)
                {
                    break;
                }
            }

            order.Clear();
        }

        foreach (var (key, entry) in entries)
        {
            if (entry.Count == 0)
            {
                entries.Remove(key);
            }
        }
    }

    private enum DecompositionKind : byte
    {
        Lu,
        Cholesky,
        Qr,
        SymmetricEigen,
        SingularValue,
    }

    private readonly record struct Key(DecompositionKind Kind, int Rows, int Columns);

    private sealed class Entry(long elements)
    {
        public readonly long Elements = elements;
        public object[] Items = [];
        public int Count;
        public int Used;
        public long LastUse;

        public void Add(object item)
        {
            if (Count == Items.Length)
            {
                Array.Resize(ref Items, Math.Max(2, 2 * Items.Length));
            }

            Items[Count++] = item;
        }

        public void RemoveLast() => Items[--Count] = null!;
    }

    private struct Active
    {
        public object Decomposition;
        public Entry Entry;
        public DecompositionInfo Info;
    }
}

// Factor/Second/Third hold borrowed factor-slot indices (-1 if unused): Cholesky(L), QR(Q, R), SymmetricEigen(Values, Vectors), SVD(Values, U, V).
internal struct DecompositionInfo
{
    public ArrayView Input;
    public int Factor;
    public int Second;
    public int Third;
    public double LogDeterminant;
    public int LogDeterminantIndex;
    public bool RequiresGradient;
}
