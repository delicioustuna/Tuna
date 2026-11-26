using System.Buffers.Binary;
using System.Security.Cryptography;

namespace Tuna.RandomSource;

/// <summary>
/// Provides a thread-safe base implementation of <see cref="IRandomSource"/> with uniform range conversion and per-call synchronization.
/// <para>一様乱数の範囲変換と呼び出し単位の同期を提供する、<see cref="IRandomSource"/> のスレッドセーフな基底クラスです。</para>
/// </summary>
/// <remarks>
/// <para>Single-value generation, bulk <c>Fill</c> operations, and state operations synchronize on the same per-instance lock. Execution order among concurrent callers is not guaranteed, and callers must ensure exclusive access to any destination span passed to <c>Fill</c>.</para>
/// <para>If an exception is thrown during generation (for example, when a counter-based stream is exhausted), values already written and state transitions already performed are not rolled back.</para>
/// <para>単発生成、一括生成（<c>Fill</c>）、および状態操作は同一のインスタンス単位のロックで同期されます。並行呼び出しの実行順序は保証されず、<c>Fill</c> に渡す出力先スパンへの排他アクセスは呼び出し側で管理する必要があります。</para>
/// <para>生成中に例外が発生した場合（カウンタ方式のストリームが枯渇した場合など）、それまでに書き込まれた値および状態の進行はロールバックされません。</para>
/// </remarks>
public abstract class RandomGenerator : IRandomSource
{
    private protected readonly Lock Sync = new();
    private protected const double DoubleUnit = 1.0 / (1UL << 53);
    private protected const float SingleUnit = 1.0f / (1U << 24);

    private protected RandomGenerator() { }

    private protected abstract uint ReadUInt32();
    private protected abstract ulong ReadUInt64();

    /// <inheritdoc/>
    public uint NextUInt32()
    {
        lock (Sync)
        {
            return ReadUInt32();
        }
    }
    /// <inheritdoc/>
    public ulong NextUInt64()
    {
        lock (Sync)
        {
            return ReadUInt64();
        }
    }

    /// <inheritdoc/>
    public uint NextUInt32(uint exclusiveMax)
    {
        ArgumentOutOfRangeException.ThrowIfZero(exclusiveMax);
        lock (Sync)
        {
            return BoundedUInt32(exclusiveMax);
        }
    }

    /// <inheritdoc/>
    public ulong NextUInt64(ulong exclusiveMax)
    {
        ArgumentOutOfRangeException.ThrowIfZero(exclusiveMax);
        lock (Sync)
        {
            return BoundedUInt64(exclusiveMax);
        }
    }

    /// <inheritdoc/>
    public int NextInt32() => NextInt32(int.MaxValue);

    /// <inheritdoc/>
    public int NextInt32(int exclusiveMax)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(exclusiveMax);
        if (exclusiveMax <= 1)
        {
            return 0;
        }

        lock (Sync)
        {
            return (int)BoundedUInt32((uint)exclusiveMax);
        }
    }

    /// <inheritdoc/>
    public int NextInt32(int inclusiveMin, int exclusiveMax)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(exclusiveMax, inclusiveMin);
        uint width = unchecked((uint)exclusiveMax - (uint)inclusiveMin);
        if (width <= 1)
        {
            return inclusiveMin;
        }

        lock (Sync)
        {
            return unchecked((int)(BoundedUInt32(width) + (uint)inclusiveMin));
        }
    }

    /// <inheritdoc/>
    public long NextInt64() => NextInt64(long.MaxValue);

    /// <inheritdoc/>
    public long NextInt64(long exclusiveMax)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(exclusiveMax);
        if (exclusiveMax <= 1)
        {
            return 0;
        }

        lock (Sync)
        {
            return (long)BoundedUInt64((ulong)exclusiveMax);
        }
    }

    /// <inheritdoc/>
    public long NextInt64(long inclusiveMin, long exclusiveMax)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(exclusiveMax, inclusiveMin);
        ulong width = unchecked((ulong)exclusiveMax - (ulong)inclusiveMin);
        if (width <= 1)
        {
            return inclusiveMin;
        }

        lock (Sync)
        {
            return unchecked((long)(BoundedUInt64(width) + (ulong)inclusiveMin));
        }
    }

    // Lemire の乗算棄却法（https://arxiv.org/abs/1805.10941）。積の下位語が 2^w mod bound 未満の入力を棄却する。
    private uint BoundedUInt32(uint bound)
    {
        if (bound == 1)
        {
            return 0;
        }

        ulong product = (ulong)ReadUInt32() * bound;
        uint remainder = unchecked((uint)product);
        if (remainder < bound)
        {
            uint cutoff = unchecked(0U - bound) % bound;
            while (remainder < cutoff)
            {
                product = (ulong)ReadUInt32() * bound;
                remainder = unchecked((uint)product);
            }
        }

        return (uint)(product >> 32);
    }

    // 同期を全体で 1 回にまとめつつ、NextInt32(int) の連続呼び出しと同一の結果および乱数消費になる Fisher–Yates シャッフル。
    internal void ShuffleSynchronized<T>(Span<T> values)
    {
        lock (Sync)
        {
            for (int i = values.Length - 1; i > 0; i--)
            {
                int j = (int)BoundedUInt32((uint)i + 1);
                (values[i], values[j]) = (values[j], values[i]);
            }
        }
    }

    private ulong BoundedUInt64(ulong bound)
    {
        if (bound == 1)
        {
            return 0;
        }

        ulong upper = Math.BigMul(ReadUInt64(), bound, out ulong lower);
        if (lower < bound)
        {
            ulong cutoff = unchecked(0UL - bound) % bound;
            while (lower < cutoff)
            {
                upper = Math.BigMul(ReadUInt64(), bound, out lower);
            }
        }

        return upper;
    }

    /// <inheritdoc/>
    public double NextDouble()
    {
        lock (Sync)
        {
            return (ReadUInt64() >> 11) * DoubleUnit;
        }
    }

    /// <inheritdoc/>
    public double NextDoubleOpen()
    {
        lock (Sync)
        {
            ulong numerator;
            do
            {
                numerator = ReadUInt64() >> 11;
            } while (numerator == 0);
            return numerator * DoubleUnit;
        }
    }

    /// <inheritdoc/>
    public float NextSingle()
    {
        lock (Sync)
        {
            return (ReadUInt32() >> 8) * SingleUnit;
        }
    }

    /// <inheritdoc/>
    public void Fill(Span<uint> destination)
    {
        lock (Sync)
        {
            FillUInt32(destination);
        }
    }

    /// <inheritdoc/>
    public void Fill(Span<ulong> destination)
    {
        lock (Sync)
        {
            FillUInt64(destination);
        }
    }

    /// <inheritdoc/>
    public void Fill(Span<double> destination)
    {
        lock (Sync)
        {
            FillDouble(destination);
        }
    }

    /// <inheritdoc/>
    public void Fill(Span<float> destination)
    {
        lock (Sync)
        {
            for (int i = 0; i < destination.Length; i++)
            {
                destination[i] = (ReadUInt32() >> 8) * SingleUnit;
            }
        }
    }

    /// <inheritdoc/>
    public void Fill(Span<byte> destination)
    {
        lock (Sync)
        {
            while (destination.Length >= sizeof(ulong))
            {
                BinaryPrimitives.WriteUInt64LittleEndian(destination, ReadUInt64());
                destination = destination[sizeof(ulong)..];
            }

            if (!destination.IsEmpty)
            {
                ulong tail = ReadUInt64();
                for (int i = 0; i < destination.Length; i++, tail >>= 8)
                {
                    destination[i] = unchecked((byte)tail);
                }
            }
        }
    }

    private protected virtual void FillUInt32(Span<uint> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = ReadUInt32();
        }
    }

    private protected virtual void FillUInt64(Span<ulong> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = ReadUInt64();
        }
    }

    private protected virtual void FillDouble(Span<double> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = (ReadUInt64() >> 11) * DoubleUnit;
        }
    }

    private protected static ulong CreateSeed()
    {
        Span<byte> bytes = stackalloc byte[sizeof(ulong)];
        RandomNumberGenerator.Fill(bytes);
        return BinaryPrimitives.ReadUInt64LittleEndian(bytes);
    }
}
