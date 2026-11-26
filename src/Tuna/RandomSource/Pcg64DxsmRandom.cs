using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;

namespace Tuna.RandomSource;

/// <summary>
/// Represents a serializable snapshot of a <see cref="Pcg64DxsmRandom"/> generator's internal state.
/// <para><see cref="Pcg64DxsmRandom"/> の内部状態を保存するためのスナップショットを表します。</para>
/// </summary>
/// <param name="State">The 128-bit LCG state value. / 128 ビットの LCG 状態値。</param>
/// <param name="Increment">The 128-bit LCG increment (must be odd). / 128 ビットの LCG 増分（奇数である必要があります）。</param>
/// <param name="Version">The state format version (currently <c>1</c>). / 状態フォーマットのバージョン（現在は <c>1</c>）。</param>
/// <remarks>
/// <para><paramref name="Increment"/> must be odd and <paramref name="Version"/> must be <c>1</c> when passed to <see cref="Pcg64DxsmRandom.RestoreState(Pcg64DxsmState)"/>.</para>
/// <para><see cref="Pcg64DxsmRandom.RestoreState(Pcg64DxsmState)"/> に渡す際は、<paramref name="Increment"/> は奇数、<paramref name="Version"/> は <c>1</c> である必要があります。</para>
/// </remarks>
public readonly record struct Pcg64DxsmState(UInt128 State, UInt128 Increment, int Version = 1);

/// <summary>
/// A thread-safe PCG64 DXSM (128-bit state, 64-bit output) pseudorandom number generator with deterministic seeding and stream selection.
/// <para>決定的なシード初期化とストリーム識別をサポートする、スレッドセーフな PCG64 DXSM（128 ビット状態・64 ビット出力）擬似乱数生成器です。</para>
/// </summary>
/// <remarks>
/// <para>Implements the PCG CM DXSM 128/64 algorithm with a period of <c>2¹²⁸</c>, applying the DXSM output permutation to the pre-update state (<see href="https://numpy.org/doc/stable/reference/random/bit_generators/pcg64dxsm.html">algorithm specification</see>).</para>
/// <para>Seeding uses a library-specific SHA-256 expansion of the seed and stream ID, so numeric seeds produce different sequences from NumPy's <c>PCG64DXSM</c>. All public members are thread-safe via per-instance locking.</para>
/// <para>周期 <c>2¹²⁸</c> の PCG CM DXSM 128/64 アルゴリズムを実装し、状態更新前の値に DXSM 出力変換を適用します（<see href="https://numpy.org/doc/stable/reference/random/bit_generators/pcg64dxsm.html">アルゴリズム仕様</see>）。</para>
/// <para>シードとストリーム ID の展開には本ライブラリ固有の SHA-256 ベースの初期化を使用するため、同一の数値シードであっても NumPy の <c>PCG64DXSM</c> とは異なる乱数列になります。公開メンバーはすべてインスタンス単位のロックによりスレッドセーフです。</para>
/// </remarks>
public sealed class Pcg64DxsmRandom : RandomGenerator
{
    private const ulong Multiplier = 0xDA942042E4DD58B5UL;
    private UInt128 state;
    private UInt128 increment;

    /// <summary>
    /// Initializes a new instance of <see cref="Pcg64DxsmRandom"/> with a seed obtained from the operating system's cryptographic random number generator.
    /// <para>OS の暗号論的乱数生成器から取得したシードで <see cref="Pcg64DxsmRandom"/> の新しいインスタンスを初期化します。</para>
    /// </summary>
    /// <remarks>
    /// <para>To reproduce the generated sequence later, capture the initial state with <see cref="GetState"/>.</para>
    /// <para>生成した乱数列を後から再現するには、<see cref="GetState"/> で初期状態を保存してください。</para>
    /// </remarks>
    public Pcg64DxsmRandom() : this(CreateSeed()) { }

    /// <summary>
    /// Initializes a new instance of <see cref="Pcg64DxsmRandom"/> from the specified seed and logical stream identifier.
    /// <para>指定したシードと論理的なストリーム識別子から <see cref="Pcg64DxsmRandom"/> の新しいインスタンスを初期化します。</para>
    /// </summary>
    /// <param name="seed">The 64-bit seed value. / 64 ビットのシード値。</param>
    /// <param name="streamId">The 64-bit logical stream identifier used to derive an independent stream. / 独立したストリームを派生させるための 64 ビット論理ストリーム識別子。</param>
    public Pcg64DxsmRandom(ulong seed, ulong streamId = 0)
    {
        // 入力の形式とドメイン文字列は再現性の契約として固定する。
        ReadOnlySpan<byte> domain = "Tuna.PCG64DXSM.v1"u8;
        Span<byte> input = stackalloc byte[domain.Length + 16];
        domain.CopyTo(input);
        BinaryPrimitives.WriteUInt64LittleEndian(input[domain.Length..], seed);
        BinaryPrimitives.WriteUInt64LittleEndian(input[(domain.Length + 8)..], streamId);
        Span<byte> digest = stackalloc byte[32];
        SHA256.HashData(input, digest);
        UInt128 initial = BinaryPrimitives.ReadUInt128LittleEndian(digest);
        increment = BinaryPrimitives.ReadUInt128LittleEndian(digest[16..]) | 1;
        // LCG を 0 から 1 回進め、初期状態を加算してからさらに 1 回進める標準初期化。
        state = unchecked((initial + increment) * Multiplier + increment);
    }

    /// <summary>
    /// Initializes a new instance of <see cref="Pcg64DxsmRandom"/> restored from a previously saved state.
    /// <para>保存された状態から <see cref="Pcg64DxsmRandom"/> の新しいインスタンスを復元します。</para>
    /// </summary>
    /// <param name="state">The saved generator state to restore. / 復元する乱数生成器の保存状態。</param>
    /// <exception cref="ArgumentException"><paramref name="state"/> has an unsupported <see cref="Pcg64DxsmState.Version"/> or an even <see cref="Pcg64DxsmState.Increment"/>. / <paramref name="state"/> の <see cref="Pcg64DxsmState.Version"/> が未対応であるか、<see cref="Pcg64DxsmState.Increment"/> が偶数の場合。</exception>
    public Pcg64DxsmRandom(Pcg64DxsmState state) => RestoreState(state);

    /// <summary>
    /// Captures a consistent snapshot of the current generator state.
    /// <para>現在の乱数生成器の状態を一貫したスナップショットとして取得します。</para>
    /// </summary>
    /// <returns>The current <see cref="Pcg64DxsmState"/>. / 現在の <see cref="Pcg64DxsmState"/>。</returns>
    public Pcg64DxsmState GetState()
    {
        lock (Sync)
        {
            return new(state, increment);
        }
    }

    /// <summary>
    /// Restores the generator to a previously saved state.
    /// <para>乱数生成器を保存された状態へ復元します。</para>
    /// </summary>
    /// <param name="saved">The saved generator state to restore. / 復元する乱数生成器の保存状態。</param>
    /// <exception cref="ArgumentException"><paramref name="saved"/> has a <see cref="Pcg64DxsmState.Version"/> other than <c>1</c> or an even <see cref="Pcg64DxsmState.Increment"/>. / <paramref name="saved"/> の <see cref="Pcg64DxsmState.Version"/> が <c>1</c> 以外であるか、<see cref="Pcg64DxsmState.Increment"/> が偶数の場合。</exception>
    public void RestoreState(Pcg64DxsmState saved)
    {
        if (saved.Version != 1)
        {
            throw new ArgumentException("Unsupported state version.", nameof(saved));
        }

        if ((saved.Increment & 1) == 0)
        {
            throw new ArgumentException("The increment must be odd.", nameof(saved));
        }

        lock (Sync)
        {
            state = saved.State;
            increment = saved.Increment;
        }
    }

    /// <summary>
    /// Advances the generator state by the specified number of raw 64-bit outputs in <c>O(log count)</c> time.
    /// <para>生の 64 ビット出力を指定個数だけ消費した位置へ <c>O(log count)</c> 時間で状態を進めます。</para>
    /// </summary>
    /// <param name="count">The number of raw 64-bit outputs to advance. / 進める生の 64 ビット出力の個数。</param>
    /// <remarks>
    /// <para>This advances the underlying 64-bit LCG step counter, which does not necessarily match the number of calls to bounded or rejection-based methods such as <see cref="RandomGenerator.NextInt32(int)"/>.</para>
    /// <para>移動量は内部の 64 ビット LCG ステップ数であり、<see cref="RandomGenerator.NextInt32(int)"/> のような棄却を伴うメソッドの呼び出し回数とは必ずしも一致しません。</para>
    /// </remarks>
    public void Advance(UInt128 count)
    {
        lock (Sync)
        {
            unchecked
            {
                // mod 2^128 のアフィン変換を繰り返し二乗法で合成し、最大 128 回の反復で進める。
                UInt128 factor = Multiplier, offset = increment;
                while (count != 0)
                {
                    if ((count & 1) != 0)
                    {
                        state = factor * state + offset;
                    }

                    offset *= factor + 1;
                    factor *= factor;
                    count >>= 1;
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private ulong TakeUInt64()
    {
        unchecked
        {
            ulong upper = (ulong)(state >> 64);
            ulong oddLower = (ulong)state | 1;
            // DXSM = ((h xor h>>32) * M xor (((h xor h>>32) * M)>>48)) * (l or 1).
            ulong folded = (upper ^ (upper >> 32)) * Multiplier;
            ulong result = (folded ^ (folded >> 48)) * oddLower;
            state = state * Multiplier + increment;
            return result;
        }
    }

    // 32 ビット要求には 64 ビット出力 1 個の上位 32 ビットを使い、下位半語のキャッシュは持たない。
    private protected override uint ReadUInt32() => (uint)(TakeUInt64() >> 32);
    private protected override ulong ReadUInt64() => TakeUInt64();

    private protected override void FillUInt32(Span<uint> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = (uint)(TakeUInt64() >> 32);
        }
    }

    private protected override void FillUInt64(Span<ulong> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = TakeUInt64();
        }
    }

    private protected override void FillDouble(Span<double> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = (TakeUInt64() >> 11) * DoubleUnit;
        }
    }
}
