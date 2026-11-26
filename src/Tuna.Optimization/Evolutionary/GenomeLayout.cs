namespace Tuna.Optimization.Evolutionary;

/// <summary>
/// Describes the structure and bounds of a mixed-type genome containing real, integer, binary, categorical, and permutation sections.
/// <para>実数・整数・二値・カテゴリ・順列の各区画からなる混合型ゲノムの構成と範囲を定義します。</para>
/// </summary>
/// <remarks>
/// <para>Sections are ordered as reals, integers, bits, categories, and permutations; any section may be empty, provided the genome contains at least one gene or permutation in total.</para>
/// <para>区画は実数、整数、二値、カテゴリ、順列の順に並びます。個々の区画は省略できますが、ゲノム全体として少なくとも1つの遺伝子または順列が必要です。</para>
/// </remarks>
public sealed class GenomeLayout
{
    /// <summary>
    /// The maximum allowed width (<c>upper − lower</c>) for an integer gene (<c>2⁵³</c>), ensuring exact <see cref="double"/> representation of offsets from the lower bound.
    /// <para>整数遺伝子の許容最大幅（<c>upper − lower</c>）である <c>2⁵³</c> を表します。この範囲内では下限からのオフセットを <see cref="double"/> で正確に表現できます。</para>
    /// </summary>
    public const ulong MaxIntegerWidth = 1UL << 53;

    private readonly double[] realLower;
    private readonly double[] realUpper;
    private readonly long[] integerLower;
    private readonly long[] integerUpper;
    private readonly int[] categoryCounts;
    private readonly int[] permutationLengths;
    private readonly int[] permutationOffsets;

    /// <summary>
    /// Initializes a new <see cref="GenomeLayout"/> from the specified section definitions.
    /// <para>指定した各区画の定義から <see cref="GenomeLayout"/> を初期化します。</para>
    /// </summary>
    /// <param name="realLowerBounds">Finite lower bounds for real genes. / 実数遺伝子の有限な下限。</param>
    /// <param name="realUpperBounds">Finite upper bounds for real genes (at least the lower bound, with finite width). Genes with equal bounds stay fixed. / 実数遺伝子の有限な上限（下限以上かつ幅が有限）。下限と等しい遺伝子は固定されます。</param>
    /// <param name="integerLowerBounds">Inclusive lower bounds for integer genes. / 整数遺伝子の下限（含む）。</param>
    /// <param name="integerUpperBounds">Inclusive upper bounds for integer genes (at least the lower bound, with width at most <see cref="MaxIntegerWidth"/>). / 整数遺伝子の上限（含む。下限以上かつ幅が <see cref="MaxIntegerWidth"/> 以下）。</param>
    /// <param name="bitCount">The number of binary genes (non-negative). / 二値遺伝子の数（0 以上）。</param>
    /// <param name="categoryCounts">The number of categories for each categorical gene (each at least 1; values lie in <c>[0, count − 1]</c>). / 各カテゴリ遺伝子のカテゴリ数（各 1 以上。値は <c>[0, count − 1]</c>）。</param>
    /// <param name="permutationLengths">The length of each permutation (each at least 1; a permutation of length <c>n</c> permutes <c>0..n−1</c>). / 各順列の長さ（各 1 以上。長さ <c>n</c> の順列は <c>0..n−1</c> の置換）。</param>
    /// <exception cref="ArgumentException">Bound lengths differ, bounds are invalid or non-finite, or the genome has no genes or permutations. / 下限と上限の長さが一致しない、範囲が不正または非有限、あるいは遺伝子も順列も1つも指定されていない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="bitCount"/> is negative, a category count or permutation length is less than 1, or the total permutation length exceeds <see cref="int.MaxValue"/>. / <paramref name="bitCount"/> が負、カテゴリ数や順列長が 1 未満、または順列長の合計が <see cref="int.MaxValue"/> を超える場合。</exception>
    public GenomeLayout(
        ReadOnlySpan<double> realLowerBounds = default, ReadOnlySpan<double> realUpperBounds = default,
        ReadOnlySpan<long> integerLowerBounds = default, ReadOnlySpan<long> integerUpperBounds = default,
        int bitCount = 0, ReadOnlySpan<int> categoryCounts = default, ReadOnlySpan<int> permutationLengths = default)
    {
        ValidateRealBounds(realLowerBounds, realUpperBounds, nameof(realLowerBounds), nameof(realUpperBounds));
        if (integerLowerBounds.Length != integerUpperBounds.Length)
        {
            throw new ArgumentException("The integer bounds must have the same length.", nameof(integerUpperBounds));
        }

        for (int d = 0; d < integerLowerBounds.Length; d++)
        {
            long lo = integerLowerBounds[d], hi = integerUpperBounds[d];
            if (lo > hi)
            {
                throw new ArgumentException($"Integer lower bound {d} must not exceed the upper bound.", nameof(integerLowerBounds));
            }

            if (unchecked((ulong)(hi - lo)) > MaxIntegerWidth)
            {
                throw new ArgumentException($"The width of integer bound {d} must not exceed 2^53.", nameof(integerUpperBounds));
            }
        }

        ArgumentOutOfRangeException.ThrowIfNegative(bitCount);
        foreach (int count in categoryCounts)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(count, 1, nameof(categoryCounts));
        }

        long totalLength = 0;
        foreach (int length in permutationLengths)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(length, 1, nameof(permutationLengths));
            totalLength += length;
        }

        if (totalLength > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(permutationLengths), "The total length of the permutations must not exceed int.MaxValue.");
        }

        if (realLowerBounds.Length + integerLowerBounds.Length + (long)bitCount + categoryCounts.Length + permutationLengths.Length == 0)
        {
            throw new ArgumentException("The genome must have at least one gene or permutation.", nameof(realLowerBounds));
        }

        realLower = realLowerBounds.ToArray();
        realUpper = realUpperBounds.ToArray();
        integerLower = integerLowerBounds.ToArray();
        integerUpper = integerUpperBounds.ToArray();
        BitCount = bitCount;
        this.categoryCounts = categoryCounts.ToArray();
        this.permutationLengths = permutationLengths.ToArray();
        permutationOffsets = new int[permutationLengths.Length];
        int offset = 0;
        for (int k = 0; k < permutationLengths.Length; k++)
        {
            permutationOffsets[k] = offset;
            offset += permutationLengths[k];
        }

        PermutationElementCount = offset;
    }

    /// <summary>
    /// Gets the number of real-valued genes.
    /// <para>実数遺伝子の数を取得します。</para>
    /// </summary>
    public int RealCount => realLower.Length;

    /// <summary>
    /// Gets the number of integer-valued genes.
    /// <para>整数遺伝子の数を取得します。</para>
    /// </summary>
    public int IntegerCount => integerLower.Length;

    /// <summary>
    /// Gets the number of binary genes.
    /// <para>二値遺伝子の数を取得します。</para>
    /// </summary>
    public int BitCount { get; }

    /// <summary>
    /// Gets the number of categorical genes.
    /// <para>カテゴリ遺伝子の数を取得します。</para>
    /// </summary>
    public int CategoryCount => categoryCounts.Length;

    /// <summary>
    /// Gets the number of permutation genes.
    /// <para>順列の数を取得します。</para>
    /// </summary>
    public int PermutationCount => permutationLengths.Length;

    /// <summary>
    /// Gets the total number of elements across all permutations.
    /// <para>すべての順列の長さの合計を取得します。</para>
    /// </summary>
    public int PermutationElementCount { get; }

    /// <summary>
    /// Gets the total number of mutation units (the sum of real, integer, binary, and categorical gene counts plus the number of permutations).
    /// <para>突然変異単位の総数（実数・整数・二値・カテゴリの各遺伝子数と順列数の合計）を取得します。既定の突然変異確率はこの逆数になります。</para>
    /// </summary>
    public long MutationUnitCount => RealCount + IntegerCount + (long)BitCount + CategoryCount + PermutationCount;

    /// <summary>
    /// Gets the lower bounds for the real-valued genes.
    /// <para>実数遺伝子の下限を取得します。</para>
    /// </summary>
    public ReadOnlySpan<double> RealLowerBounds => realLower;

    /// <summary>
    /// Gets the upper bounds for the real-valued genes.
    /// <para>実数遺伝子の上限を取得します。</para>
    /// </summary>
    public ReadOnlySpan<double> RealUpperBounds => realUpper;

    /// <summary>
    /// Gets the inclusive lower bounds for the integer-valued genes.
    /// <para>整数遺伝子の下限を取得します。</para>
    /// </summary>
    public ReadOnlySpan<long> IntegerLowerBounds => integerLower;

    /// <summary>
    /// Gets the inclusive upper bounds for the integer-valued genes.
    /// <para>整数遺伝子の上限を取得します。</para>
    /// </summary>
    public ReadOnlySpan<long> IntegerUpperBounds => integerUpper;

    /// <summary>
    /// Gets the number of categories for each categorical gene.
    /// <para>各カテゴリ遺伝子のカテゴリ数を取得します。</para>
    /// </summary>
    public ReadOnlySpan<int> CategoryCounts => categoryCounts;

    /// <summary>
    /// Gets the length of each permutation.
    /// <para>各順列の長さを取得します。</para>
    /// </summary>
    public ReadOnlySpan<int> PermutationLengths => permutationLengths;

    /// <summary>
    /// Gets the starting offset of the specified permutation within the concatenated permutation buffer.
    /// <para>連結された順列配列内における指定した順列の開始オフセットを取得します。</para>
    /// </summary>
    /// <param name="index">The zero-based permutation index. / 0 から始まる順列インデックス。</param>
    /// <returns>The zero-based starting offset of the permutation. / 順列の先頭要素の 0 始まりオフセット。</returns>
    public int PermutationOffset(int index) => permutationOffsets[index];

    internal static void ValidateRealBounds(ReadOnlySpan<double> lower, ReadOnlySpan<double> upper, string lowerName, string upperName)
    {
        if (lower.Length != upper.Length)
        {
            throw new ArgumentException("The bounds must have the same length.", upperName);
        }

        for (int d = 0; d < lower.Length; d++)
        {
            double lo = lower[d], hi = upper[d];
            if (!double.IsFinite(lo) || !double.IsFinite(hi))
            {
                throw new ArgumentException($"Bound {d} must be finite.", double.IsFinite(lo) ? upperName : lowerName);
            }

            if (lo > hi)
            {
                throw new ArgumentException($"Lower bound {d} must not exceed the upper bound.", lowerName);
            }

            if (!double.IsFinite(hi - lo))
            {
                throw new ArgumentException($"The width of bound {d} must be finite.", upperName);
            }
        }
    }
}
