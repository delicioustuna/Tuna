using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Tuna.RandomSource;

/// <summary>
/// Represents a 4-word (128-bit) output block produced by Philox4x32-10.
/// <para>Philox4x32-10 が生成する 4 語（128 ビット）の出力ブロックを表します。</para>
/// </summary>
/// <param name="Word0">The first 32-bit output word (index 0). / 1 番目の 32 ビット出力語（インデックス 0）。</param>
/// <param name="Word1">The second 32-bit output word (index 1). / 2 番目の 32 ビット出力語（インデックス 1）。</param>
/// <param name="Word2">The third 32-bit output word (index 2). / 3 番目の 32 ビット出力語（インデックス 2）。</param>
/// <param name="Word3">The fourth 32-bit output word (index 3). / 4 番目の 32 ビット出力語（インデックス 3）。</param>
public readonly record struct Philox4x32Block(uint Word0, uint Word1, uint Word2, uint Word3)
{
    /// <summary>
    /// Gets the 32-bit output word at the specified index within the block.
    /// <para>ブロック内の指定したインデックスにある 32 ビット出力語を取得します。</para>
    /// </summary>
    /// <param name="index">The zero-based word index (0 through 3). / 0 から始まる語インデックス（0 以上 3 以下）。</param>
    /// <returns>The 32-bit output word at <paramref name="index"/>. / <paramref name="index"/> の位置にある 32 ビット出力語。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is less than 0 or greater than 3. / <paramref name="index"/> が 0 未満または 3 より大きい場合。</exception>
    public uint this[int index] => index switch
    {
        0 => Word0,
        1 => Word1,
        2 => Word2,
        3 => Word3,
        _ => throw new ArgumentOutOfRangeException(nameof(index))
    };
}

/// <summary>
/// Represents a serializable snapshot of a <see cref="Philox4x32Random"/> generator's state.
/// <para><see cref="Philox4x32Random"/> の状態を保存するためのスナップショットを表します。</para>
/// </summary>
/// <param name="Seed">The 64-bit seed (Philox key). / 64 ビットのシード（Philox のキー）。</param>
/// <param name="StreamId">The 64-bit stream identifier (upper 64 bits of the 128-bit counter). / 64 ビットのストリーム識別子（128 ビットカウンタの上位 64 ビット）。</param>
/// <param name="BlockIndex">The zero-based 64-bit block index (lower 64 bits of the 128-bit counter). / 0 から始まる 64 ビットのブロック番号（128 ビットカウンタの下位 64 ビット）。</param>
/// <param name="WordIndex">The next unread word index in the current block (0 through 4, where 4 indicates the current block is fully consumed). / 現在のブロック内で次に読み出す語の位置（0 以上 4 以下。4 は現在のブロックを消費済みであることを表します）。</param>
/// <param name="Version">The state format version (currently <c>1</c>). / 状態フォーマットのバージョン（現在は <c>1</c>）。</param>
public readonly record struct Philox4x32State(
    ulong Seed, ulong StreamId, ulong BlockIndex, int WordIndex, int Version = 1);

/// <summary>
/// A thread-safe Philox4x32-10 counter-based pseudorandom number generator with 64-bit seed and stream identifiers.
/// <para>64 ビットのシードとストリーム識別子を持つ、スレッドセーフなカウンタ方式擬似乱数生成器（Philox4x32-10）です。</para>
/// </summary>
/// <remarks>
/// <para>Implements the 10-round Philox4x32 algorithm (<see href="https://www.thesalmons.org/john/random123/papers/random123sc11.pdf">Salmon et al., 2011</see>). The 64-bit key is the seed, the upper 64 bits of the 128-bit counter hold the stream ID, and the lower 64 bits hold the block index.</para>
/// <para>Each stream produces up to <c>2⁶⁴</c> blocks (four 32-bit words per block). Attempting to read past the end of a stream throws <see cref="InvalidOperationException"/>. Identical seeds, stream IDs, and call sequences always yield identical outputs. All public members are thread-safe via per-instance locking.</para>
/// <para>10 ラウンドの Philox4x32 アルゴリズム（<see href="https://www.thesalmons.org/john/random123/papers/random123sc11.pdf">Salmon et al., 2011</see>）を実装します。64 ビットキーにシードを、128 ビットカウンタの上位 64 ビットにストリーム ID、下位 64 ビットにブロック番号を割り当てます。</para>
/// <para>各ストリームは最大 <c>2⁶⁴</c> ブロック（1 ブロックあたり 32 ビット語 4 個）を生成でき、ストリーム末尾を超えて読み出そうとすると <see cref="InvalidOperationException"/> をスローします。同じシード・ストリーム ID・呼び出し列に対しては常に同一の出力を再現します。公開メンバーはすべてインスタンス単位のロックによりスレッドセーフです。</para>
/// </remarks>
public sealed class Philox4x32Random : RandomGenerator
{
    private ulong seed;
    private ulong streamId;
    private ulong blockIndex;
    private int wordIndex;
    private Philox4x32Block block;

    /// <summary>
    /// Initializes a new instance of <see cref="Philox4x32Random"/> with a seed obtained from the operating system's cryptographic random number generator.
    /// <para>OS の暗号論的乱数生成器から取得したシードで <see cref="Philox4x32Random"/> の新しいインスタンスを初期化します。</para>
    /// </summary>
    /// <remarks>
    /// <para>To reproduce the generated sequence later, capture the initial state with <see cref="GetState"/>.</para>
    /// <para>生成した乱数列を後から再現するには、<see cref="GetState"/> で初期状態を保存してください。</para>
    /// </remarks>
    public Philox4x32Random() : this(CreateSeed()) { }

    /// <summary>
    /// Initializes a new instance of <see cref="Philox4x32Random"/> starting at block 0, word 0 for the specified seed and stream identifier.
    /// <para>指定したシードとストリーム識別子で、ブロック 0・語 0 から開始する <see cref="Philox4x32Random"/> の新しいインスタンスを初期化します。</para>
    /// </summary>
    /// <param name="seed">The 64-bit seed value. / 64 ビットのシード値。</param>
    /// <param name="streamId">The 64-bit stream identifier. / 64 ビットのストリーム識別子。</param>
    public Philox4x32Random(ulong seed, ulong streamId = 0)
        : this(new Philox4x32State(seed, streamId, 0, 0)) { }

    /// <summary>
    /// Initializes a new instance of <see cref="Philox4x32Random"/> restored from a previously saved state.
    /// <para>保存された状態から <see cref="Philox4x32Random"/> の新しいインスタンスを復元します。</para>
    /// </summary>
    /// <param name="state">The saved generator state to restore. / 復元する乱数生成器の保存状態。</param>
    /// <exception cref="ArgumentException"><paramref name="state"/> has an unsupported <see cref="Philox4x32State.Version"/>. / <paramref name="state"/> の <see cref="Philox4x32State.Version"/> が未対応の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="state"/> has a <see cref="Philox4x32State.WordIndex"/> outside the range 0 through 4. / <paramref name="state"/> の <see cref="Philox4x32State.WordIndex"/> が 0 以上 4 以下の範囲外の場合。</exception>
    public Philox4x32Random(Philox4x32State state) => RestoreState(state);

    /// <summary>
    /// Computes the output block at the specified counter position without side effects, shared state, or locking.
    /// <para>副作用・共有状態・ロックなしで、指定したカウンタ位置の出力ブロックを計算します。</para>
    /// </summary>
    /// <param name="seed">The 64-bit seed value (Philox key). / 64 ビットのシード値（Philox のキー）。</param>
    /// <param name="streamId">The 64-bit stream identifier. / 64 ビットのストリーム識別子。</param>
    /// <param name="blockIndex">The zero-based 64-bit block index. / 0 から始まる 64 ビットのブロック番号。</param>
    /// <returns>The 4-word output block for the specified seed, stream, and block index. / 指定したシード、ストリーム、ブロック番号に対応する 4 語の出力ブロック。</returns>
    public static Philox4x32Block GenerateBlock(ulong seed, ulong streamId, ulong blockIndex)
    {
        // Philox-4x32 のラウンド関数: 2 組の 32x32 ビット乗算と上位語の交換を行う（カウンタ・キーとも下位語から配置）。
        unchecked
        {
            uint a = (uint)blockIndex, b = (uint)(blockIndex >> 32);
            uint c = (uint)streamId, d = (uint)(streamId >> 32);
            uint keyLeft = (uint)seed, keyRight = (uint)(seed >> 32);
            for (uint round = 0; round < 10; round++)
            {
                ulong leftProduct = 0xD2511F53UL * a;
                ulong rightProduct = 0xCD9E8D57UL * c;
                uint newA = (uint)(rightProduct >> 32) ^ b ^ (keyLeft + round * 0x9E3779B9U);
                uint newC = (uint)(leftProduct >> 32) ^ d ^ (keyRight + round * 0xBB67AE85U);
                (a, b, c, d) = (newA, (uint)rightProduct, newC, (uint)leftProduct);
            }

            return new(a, b, c, d);
        }
    }

    /// <summary>
    /// Captures a consistent snapshot of the current generator state, including the unread word offset.
    /// <para>未消費の語位置を含む現在の乱数生成器の状態を一貫したスナップショットとして取得します。</para>
    /// </summary>
    /// <returns>The current <see cref="Philox4x32State"/>. / 現在の <see cref="Philox4x32State"/>。</returns>
    public Philox4x32State GetState()
    {
        lock (Sync)
        {
            return new(seed, streamId, blockIndex, wordIndex);
        }
    }

    /// <summary>
    /// Restores the generator to a previously saved state.
    /// <para>乱数生成器を保存された状態へ復元します。</para>
    /// </summary>
    /// <param name="state">The saved generator state to restore. / 復元する乱数生成器の保存状態。</param>
    /// <exception cref="ArgumentException"><paramref name="state"/> has a <see cref="Philox4x32State.Version"/> other than <c>1</c>. / <paramref name="state"/> の <see cref="Philox4x32State.Version"/> が <c>1</c> 以外の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="state"/> has a <see cref="Philox4x32State.WordIndex"/> outside the range 0 through 4. / <paramref name="state"/> の <see cref="Philox4x32State.WordIndex"/> が 0 以上 4 以下の範囲外の場合。</exception>
    public void RestoreState(Philox4x32State state)
    {
        if (state.Version != 1)
        {
            throw new ArgumentException("Unsupported state version.", nameof(state));
        }

        if (state.WordIndex < 0 || state.WordIndex > 4)
        {
            throw new ArgumentOutOfRangeException(nameof(state));
        }

        lock (Sync)
        {
            seed = state.Seed;
            streamId = state.StreamId;
            blockIndex = state.BlockIndex;
            wordIndex = state.WordIndex;
            block = GenerateBlock(seed, streamId, blockIndex);
        }
    }

    /// <summary>
    /// Moves the generator to the specified block and word position within the current seed and stream.
    /// <para>現在のシードとストリームのまま、指定したブロック番号および語位置へ移動します。</para>
    /// </summary>
    /// <param name="blockIndex">The zero-based 64-bit block index. / 0 から始まる 64 ビットのブロック番号。</param>
    /// <param name="wordIndex">The zero-based word index within the block (0 through 3). / ブロック内の 0 から始まる語位置（0 以上 3 以下）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="wordIndex"/> is less than 0 or greater than 3. / <paramref name="wordIndex"/> が 0 未満または 3 より大きい場合。</exception>
    public void Seek(ulong blockIndex, int wordIndex = 0)
    {
        if (wordIndex < 0 || wordIndex >= 4)
        {
            throw new ArgumentOutOfRangeException(nameof(wordIndex));
        }

        lock (Sync)
        {
            this.blockIndex = blockIndex;
            this.wordIndex = wordIndex;
            block = GenerateBlock(seed, streamId, blockIndex);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private uint TakeWord()
    {
        if (wordIndex == 4)
        {
            if (blockIndex == ulong.MaxValue)
            {
                throw new InvalidOperationException("The Philox stream is exhausted.");
            }

            block = GenerateBlock(seed, streamId, ++blockIndex);
            wordIndex = 0;
        }

        return block[wordIndex++];
    }

    private protected override uint ReadUInt32() => TakeWord();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private ulong TakeUInt64()
    {
        // ストリーム末尾に 1 語しか残っていない場合、失敗した 64 ビット読み出しでその語を消費しない。
        if (blockIndex == ulong.MaxValue && wordIndex >= 3)
        {
            throw new InvalidOperationException("Fewer than two words remain in the Philox stream.");
        }

        ulong lower = TakeWord();
        return lower | ((ulong)TakeWord() << 32);
    }

    private protected override ulong ReadUInt64() => TakeUInt64();

    private protected override void FillUInt32(Span<uint> destination)
    {
        int i = 0;
        while (i < destination.Length && wordIndex < 4)
        {
            destination[i++] = TakeWord();
        }

        if (i < destination.Length)
        {
            int blocks = AvailableBlocks((destination.Length - i) / 4);
            GenerateBlocks(destination.Slice(i, 4 * blocks));
            i += 4 * blocks;
        }

        while (i < destination.Length)
        {
            destination[i++] = TakeWord();
        }
    }

    private protected override void FillUInt64(Span<ulong> destination)
    {
        int i = 0;
        while (i < destination.Length && wordIndex != 4 && wordIndex != 3)
        {
            destination[i++] = TakeUInt64();
        }

        i += FillPairs(destination[i..]);
        while (i < destination.Length)
        {
            destination[i++] = TakeUInt64();
        }
    }

    private protected override void FillDouble(Span<double> destination)
    {
        int i = 0;
        while (i < destination.Length && wordIndex != 4 && wordIndex != 3)
        {
            destination[i++] = (TakeUInt64() >> 11) * DoubleUnit;
        }

        // 64 ビット整数を出力先スパンに直接生成してから、インプレースで倍精度浮動小数点数へ変換する。
        var bits = MemoryMarshal.Cast<double, ulong>(destination[i..]);
        int filled = FillPairs(bits);
        for (int j = 0; j < filled; j++)
        {
            destination[i + j] = (bits[j] >> 11) * DoubleUnit;
        }

        i += filled;
        while (i < destination.Length)
        {
            destination[i++] = (TakeUInt64() >> 11) * DoubleUnit;
        }
    }

    // wordIndex が 4（ブロック境界）または 3 のときに後続の語ペアをブロック単位で書き込む。
    // 語数（要素数の 2 倍）が int に収まるよう MaxPairs ごとに分割し、末尾と枯渇の判定は呼び出し元の 1 要素経路へ委ねる。
    private int FillPairs(Span<ulong> destination)
    {
        int filled = 0;
        while (filled < destination.Length)
        {
            int written = FillPairsPiece(destination.Slice(filled, Math.Min(destination.Length - filled, MaxPairs)));
            if (written == 0)
            {
                break;
            }

            filled += written;
        }

        return filled;
    }

    // 語数（要素数の 2 倍）が int に収まるための 1 回あたりの上限。
    private const int MaxPairs = 1 << 28;

    private int FillPairsPiece(Span<ulong> destination)
    {
        if (destination.IsEmpty || !BitConverter.IsLittleEndian)
        {
            return 0;
        }

        var words = MemoryMarshal.Cast<ulong, uint>(destination);
        if (wordIndex == 4)
        {
            int blocks = AvailableBlocks(destination.Length / 2);
            GenerateBlocks(words[..(4 * blocks)]);
            return 2 * blocks;
        }

        // 語 3 から始まるペアは残余の語 3 と次ブロックの語 0 で構成し、最終ブロックの語 3 を次回の繰り越しとして残す。
        // 書き込み範囲直後の要素下位を一時的に上書きするため元の値を退避・復元し、枯渇例外時に未生成要素を変更しない。
        int count = AvailableBlocks((2 * destination.Length - 1) / 4);
        if (count == 0)
        {
            return 0;
        }

        uint carry = block.Word3;
        uint next = words[4 * count];
        GenerateBlocks(words.Slice(1, 4 * count));
        words[0] = carry;
        words[4 * count] = next;
        wordIndex = 3;
        return 2 * count;
    }

    private int AvailableBlocks(int wanted) => (int)Math.Min((ulong)wanted, ulong.MaxValue - blockIndex);

    private void GenerateBlocks(Span<uint> words)
    {
        int count = words.Length / 4;
        if (count == 0)
        {
            return;
        }

        ulong index = blockIndex;
        int i = 0;
        if (Avx2.IsSupported && count >= 4)
        {
            i = count & ~3;
            GenerateBlocks4(seed, streamId, index + 1, words[..(4 * i)]);
            index += (ulong)i;
        }

        for (; i < count; i++)
        {
            var next = GenerateBlock(seed, streamId, ++index);
            words[4 * i] = next.Word0;
            words[4 * i + 1] = next.Word1;
            words[4 * i + 2] = next.Word2;
            words[4 * i + 3] = next.Word3;
        }

        blockIndex = index;
        block = new(words[4 * count - 4], words[4 * count - 3], words[4 * count - 2], words[4 * count - 1]);
        wordIndex = 4;
    }

    // AVX2 の 4 レーン（1 レーン = 1 ブロック）で 4 ブロックずつ並列生成する。
    // 32 ビット乗算は 64 ビットレーンの下位 32 ビット同士の積（vpmuludq）で計算する。
    private static void GenerateBlocks4(ulong seed, ulong streamId, ulong first, Span<uint> words)
    {
        var m0 = Vector256.Create(0xD2511F53UL).AsUInt32();
        var m1 = Vector256.Create(0xCD9E8D57UL).AsUInt32();
        var low = Vector256.Create(0xFFFFFFFFUL);
        var keyLeft = Vector256.Create((ulong)(uint)seed);
        var keyRight = Vector256.Create(seed >> 32);
        var stepLeft = Vector256.Create((ulong)0x9E3779B9U);
        var stepRight = Vector256.Create((ulong)0xBB67AE85U);
        var c0 = Vector256.Create((ulong)(uint)streamId);
        var d0 = Vector256.Create(streamId >> 32);
        var lanes = Vector256.Create(0UL, 1, 2, 3);
        ref uint destination = ref MemoryMarshal.GetReference(words);
        // 長さが int.MaxValue に近い場合でも g + 16 がオーバーフローしない終了条件にする。
        for (int g = 0; g <= words.Length - 16; g += 16, first += 4)
        {
            var a = Vector256.Create(first) + lanes;
            var b = Vector256.ShiftRightLogical(a, 32);
            var c = c0;
            var d = d0;
            var kl = keyLeft;
            var kr = keyRight;
            for (int round = 0; round < 10; round++)
            {
                var leftProduct = Avx2.Multiply(a.AsUInt32(), m0);
                var rightProduct = Avx2.Multiply(c.AsUInt32(), m1);
                a = Vector256.ShiftRightLogical(rightProduct, 32) ^ b ^ kl;
                c = Vector256.ShiftRightLogical(leftProduct, 32) ^ d ^ kr;
                b = rightProduct;
                d = leftProduct;
                kl += stepLeft;
                kr += stepRight;
            }

            // 各レーンの (Word0, Word1) と (Word2, Word3) を 64 ビットにパックし、ブロック順に転置して書き込む。
            var x = (a & low) | Vector256.ShiftLeft(b, 32);
            var y = (c & low) | Vector256.ShiftLeft(d, 32);
            var even = Avx2.UnpackLow(x, y);
            var odd = Avx2.UnpackHigh(x, y);
            Avx2.Permute2x128(even, odd, 0x20).AsUInt32().StoreUnsafe(ref destination, (nuint)g);
            Avx2.Permute2x128(even, odd, 0x31).AsUInt32().StoreUnsafe(ref destination, (nuint)(g + 8));
        }
    }
}
