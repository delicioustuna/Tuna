namespace Tuna.RandomSource;

/// <summary>
/// Represents a thread-safe uniform pseudorandom number generator.
/// <para>スレッドセーフな一様擬似乱数生成器を表します。</para>
/// </summary>
/// <remarks>
/// <para>Implementations of this interface are not suitable for cryptographic purposes.</para>
/// <para>For reproducible parallel computations, create a separate random source per logical work unit rather than sharing a single instance across threads.</para>
/// <para>このインターフェースの実装は暗号用途には適していません。</para>
/// <para>再現性が必要な並列計算では、単一のインスタンスをスレッド間で共有せず、論理的な計算単位ごとに独立した乱数源を作成してください。</para>
/// </remarks>
public interface IRandomSource
{
    /// <summary>
    /// Returns a random 32-bit unsigned integer.
    /// <para>32 ビットの符号なし一様乱数を返します。</para>
    /// </summary>
    /// <returns>A 32-bit unsigned integer across the full range of <see cref="uint"/>. / <see cref="uint"/> の全範囲にわたる 32 ビット符号なし整数。</returns>
    uint NextUInt32();

    /// <summary>
    /// Returns a random 64-bit unsigned integer.
    /// <para>64 ビットの符号なし一様乱数を返します。</para>
    /// </summary>
    /// <returns>A 64-bit unsigned integer across the full range of <see cref="ulong"/>. / <see cref="ulong"/> の全範囲にわたる 64 ビット符号なし整数。</returns>
    ulong NextUInt64();

    /// <summary>
    /// Returns a random 32-bit unsigned integer that is less than the specified maximum.
    /// <para>指定した上限未満の 32 ビット符号なし一様乱数を返します。</para>
    /// </summary>
    /// <param name="exclusiveMax">The exclusive upper bound of the random number to generate. Must be greater than 0. / 生成する乱数の排他的上限（0 より大きい値）。</param>
    /// <returns>A 32-bit unsigned integer in the range <c>[0, exclusiveMax)</c>. / <c>[0, exclusiveMax)</c> の範囲の 32 ビット符号なし整数。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="exclusiveMax"/> is 0. / <paramref name="exclusiveMax"/> が 0 の場合。</exception>
    uint NextUInt32(uint exclusiveMax);

    /// <summary>
    /// Returns a random 64-bit unsigned integer that is less than the specified maximum.
    /// <para>指定した上限未満の 64 ビット符号なし一様乱数を返します。</para>
    /// </summary>
    /// <param name="exclusiveMax">The exclusive upper bound of the random number to generate. Must be greater than 0. / 生成する乱数の排他的上限（0 より大きい値）。</param>
    /// <returns>A 64-bit unsigned integer in the range <c>[0, exclusiveMax)</c>. / <c>[0, exclusiveMax)</c> の範囲の 64 ビット符号なし整数。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="exclusiveMax"/> is 0. / <paramref name="exclusiveMax"/> が 0 の場合。</exception>
    ulong NextUInt64(ulong exclusiveMax);

    /// <summary>
    /// Returns a non-negative random 32-bit signed integer.
    /// <para>非負の 32 ビット符号付き一様乱数を返します。</para>
    /// </summary>
    /// <returns>A 32-bit signed integer in the range <c>[0, int.MaxValue)</c>. / <c>[0, int.MaxValue)</c> の範囲の 32 ビット符号付き整数。</returns>
    int NextInt32();

    /// <summary>
    /// Returns a non-negative random 32-bit signed integer that is less than the specified maximum.
    /// <para>指定した上限未満の非負 32 ビット符号付き一様乱数を返します。</para>
    /// </summary>
    /// <param name="exclusiveMax">The exclusive upper bound of the random number to generate. Must be greater than or equal to 0. / 生成する乱数の排他的上限（0 以上）。</param>
    /// <returns>A 32-bit signed integer in the range <c>[0, exclusiveMax)</c>, or 0 if <paramref name="exclusiveMax"/> is 0. / <c>[0, exclusiveMax)</c> の範囲の 32 ビット符号付き整数。<paramref name="exclusiveMax"/> が 0 の場合は 0。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="exclusiveMax"/> is less than 0. / <paramref name="exclusiveMax"/> が負の場合。</exception>
    int NextInt32(int exclusiveMax);

    /// <summary>
    /// Returns a random 32-bit signed integer within the specified range.
    /// <para>指定した範囲内の 32 ビット符号付き一様乱数を返します。</para>
    /// </summary>
    /// <param name="inclusiveMin">The inclusive lower bound of the random number returned. / 返される乱数の包括的下限。</param>
    /// <param name="exclusiveMax">The exclusive upper bound of the random number returned. Must be greater than or equal to <paramref name="inclusiveMin"/>. / 返される乱数の排他的上限（<paramref name="inclusiveMin"/> 以上）。</param>
    /// <returns>A 32-bit signed integer in the range <c>[inclusiveMin, exclusiveMax)</c>, or <paramref name="inclusiveMin"/> if both bounds are equal. / <c>[inclusiveMin, exclusiveMax)</c> の範囲の 32 ビット符号付き整数。上下限が等しい場合は <paramref name="inclusiveMin"/>。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="exclusiveMax"/> is less than <paramref name="inclusiveMin"/>. / <paramref name="exclusiveMax"/> が <paramref name="inclusiveMin"/> より小さい場合。</exception>
    int NextInt32(int inclusiveMin, int exclusiveMax);

    /// <summary>
    /// Returns a non-negative random 64-bit signed integer.
    /// <para>非負の 64 ビット符号付き一様乱数を返します。</para>
    /// </summary>
    /// <returns>A 64-bit signed integer in the range <c>[0, long.MaxValue)</c>. / <c>[0, long.MaxValue)</c> の範囲の 64 ビット符号付き整数。</returns>
    long NextInt64();

    /// <summary>
    /// Returns a non-negative random 64-bit signed integer that is less than the specified maximum.
    /// <para>指定した上限未満の非負 64 ビット符号付き一様乱数を返します。</para>
    /// </summary>
    /// <param name="exclusiveMax">The exclusive upper bound of the random number to generate. Must be greater than or equal to 0. / 生成する乱数の排他的上限（0 以上）。</param>
    /// <returns>A 64-bit signed integer in the range <c>[0, exclusiveMax)</c>, or 0 if <paramref name="exclusiveMax"/> is 0. / <c>[0, exclusiveMax)</c> の範囲の 64 ビット符号付き整数。<paramref name="exclusiveMax"/> が 0 の場合は 0。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="exclusiveMax"/> is less than 0. / <paramref name="exclusiveMax"/> が負の場合。</exception>
    long NextInt64(long exclusiveMax);

    /// <summary>
    /// Returns a random 64-bit signed integer within the specified range.
    /// <para>指定した範囲内の 64 ビット符号付き一様乱数を返します。</para>
    /// </summary>
    /// <param name="inclusiveMin">The inclusive lower bound of the random number returned. / 返される乱数の包括的下限。</param>
    /// <param name="exclusiveMax">The exclusive upper bound of the random number returned. Must be greater than or equal to <paramref name="inclusiveMin"/>. / 返される乱数の排他的上限（<paramref name="inclusiveMin"/> 以上）。</param>
    /// <returns>A 64-bit signed integer in the range <c>[inclusiveMin, exclusiveMax)</c>, or <paramref name="inclusiveMin"/> if both bounds are equal. / <c>[inclusiveMin, exclusiveMax)</c> の範囲の 64 ビット符号付き整数。上下限が等しい場合は <paramref name="inclusiveMin"/>。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="exclusiveMax"/> is less than <paramref name="inclusiveMin"/>. / <paramref name="exclusiveMax"/> が <paramref name="inclusiveMin"/> より小さい場合。</exception>
    long NextInt64(long inclusiveMin, long exclusiveMax);

    /// <summary>
    /// Returns a random double-precision floating-point number in the half-open interval <c>[0, 1)</c>.
    /// <para>半開区間 <c>[0, 1)</c> の倍精度浮動小数点一様乱数を返します。</para>
    /// </summary>
    /// <returns>A double-precision floating-point number of the form <c>k / 2⁵³</c> with <c>0 ≤ k &lt; 2⁵³</c> (step size <c>2⁻⁵³</c>). / <c>k / 2⁵³</c>（<c>0 ≤ k &lt; 2⁵³</c>、刻み幅 <c>2⁻⁵³</c>）の倍精度浮動小数点数。</returns>
    double NextDouble();

    /// <summary>
    /// Returns a random double-precision floating-point number in the open interval <c>(0, 1)</c>.
    /// <para>開区間 <c>(0, 1)</c> の倍精度浮動小数点一様乱数を返します。</para>
    /// </summary>
    /// <returns>A double-precision floating-point number of the form <c>k / 2⁵³</c> with <c>1 ≤ k &lt; 2⁵³</c>. / <c>k / 2⁵³</c>（<c>1 ≤ k &lt; 2⁵³</c>）の倍精度浮動小数点数。</returns>
    double NextDoubleOpen();

    /// <summary>
    /// Returns a random single-precision floating-point number in the half-open interval <c>[0, 1)</c>.
    /// <para>半開区間 <c>[0, 1)</c> の単精度浮動小数点一様乱数を返します。</para>
    /// </summary>
    /// <returns>A single-precision floating-point number of the form <c>k / 2²⁴</c> with <c>0 ≤ k &lt; 2²⁴</c> (step size <c>2⁻²⁴</c>). / <c>k / 2²⁴</c>（<c>0 ≤ k &lt; 2²⁴</c>、刻み幅 <c>2⁻²⁴</c>）の単精度浮動小数点数。</returns>
    float NextSingle();

    /// <summary>
    /// Fills the specified span with random 32-bit unsigned integers in the same order as repeated calls to <see cref="NextUInt32()"/>.
    /// <para>指定したスパンを、<see cref="NextUInt32()"/> の連続呼び出しと同じ順序の 32 ビット符号なし一様乱数で埋めます。</para>
    /// </summary>
    /// <param name="destination">The span to fill with random values. / 乱数を書き込む出力先スパン。</param>
    void Fill(Span<uint> destination);

    /// <summary>
    /// Fills the specified span with random 64-bit unsigned integers in the same order as repeated calls to <see cref="NextUInt64()"/>.
    /// <para>指定したスパンを、<see cref="NextUInt64()"/> の連続呼び出しと同じ順序の 64 ビット符号なし一様乱数で埋めます。</para>
    /// </summary>
    /// <param name="destination">The span to fill with random values. / 乱数を書き込む出力先スパン。</param>
    void Fill(Span<ulong> destination);

    /// <summary>
    /// Fills the specified span with random double-precision floating-point numbers in the same order as repeated calls to <see cref="NextDouble"/>.
    /// <para>指定したスパンを、<see cref="NextDouble"/> の連続呼び出しと同じ順序の倍精度浮動小数点一様乱数で埋めます。</para>
    /// </summary>
    /// <param name="destination">The span to fill with random values. / 乱数を書き込む出力先スパン。</param>
    void Fill(Span<double> destination);

    /// <summary>
    /// Fills the specified span with random single-precision floating-point numbers in the same order as repeated calls to <see cref="NextSingle"/>.
    /// <para>指定したスパンを、<see cref="NextSingle"/> の連続呼び出しと同じ順序の単精度浮動小数点一様乱数で埋めます。</para>
    /// </summary>
    /// <param name="destination">The span to fill with random values. / 乱数を書き込む出力先スパン。</param>
    void Fill(Span<float> destination);

    /// <summary>
    /// Fills the specified span with random bytes generated from 64-bit outputs in little-endian order.
    /// <para>64 ビット出力をリトルエンディアン順に展開して、指定したスパンを乱数バイトで埋めます。</para>
    /// </summary>
    /// <param name="destination">The span to fill with random bytes. / 乱数バイトを書き込む出力先スパン。</param>
    /// <remarks>
    /// <para>If the length of <paramref name="destination"/> is not a multiple of 8, any unused trailing bytes from the final 64-bit word are discarded.</para>
    /// <para><paramref name="destination"/> の長さが 8 の倍数でない場合、末尾の 64 ビット出力のうち未使用のバイトは破棄されます。</para>
    /// </remarks>
    void Fill(Span<byte> destination);
}
