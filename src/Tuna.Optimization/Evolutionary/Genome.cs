using Tuna.LinearAlgebra;

namespace Tuna.Optimization.Evolutionary;

/// <summary>
/// Provides a read-only view of a single individual's genome partitioned according to a <see cref="GenomeLayout"/>.
/// <para><see cref="GenomeLayout"/> に従って区画分けされた1個体分のゲノムへの読み取り専用ビューを表します。</para>
/// </summary>
/// <remarks>
/// <para>Views returned by <see cref="GeneticAlgorithm"/> are valid only until the next call to <see cref="GeneticAlgorithm.Tell"/>; call <see cref="ToGenome"/> to retain an independent copy.</para>
/// <para><see cref="GeneticAlgorithm"/> が返すビューは次の <see cref="GeneticAlgorithm.Tell"/> 呼び出しまで有効です。保持する場合は <see cref="ToGenome"/> で独立したコピーを作成してください。</para>
/// </remarks>
public readonly ref struct GenomeView
{
    private readonly ReadOnlySpan<int> permutations;

    internal GenomeView(GenomeLayout layout, ReadOnlySpan<double> reals, ReadOnlySpan<long> integers, ReadOnlySpan<bool> bits,
        ReadOnlySpan<int> categories, ReadOnlySpan<int> permutations)
    {
        Layout = layout;
        Reals = reals;
        Integers = integers;
        Bits = bits;
        Categories = categories;
        this.permutations = permutations;
    }

    /// <summary>
    /// Gets the layout describing this genome.
    /// <para>このゲノムの構成を取得します。</para>
    /// </summary>
    public GenomeLayout Layout { get; }

    /// <summary>
    /// Gets the real-valued genes (length <see cref="GenomeLayout.RealCount"/>).
    /// <para>実数遺伝子（長さ <see cref="GenomeLayout.RealCount"/>）を取得します。</para>
    /// </summary>
    public ReadOnlySpan<double> Reals { get; }

    /// <summary>
    /// Gets the integer-valued genes (length <see cref="GenomeLayout.IntegerCount"/>).
    /// <para>整数遺伝子（長さ <see cref="GenomeLayout.IntegerCount"/>）を取得します。</para>
    /// </summary>
    public ReadOnlySpan<long> Integers { get; }

    /// <summary>
    /// Gets the binary genes (length <see cref="GenomeLayout.BitCount"/>).
    /// <para>二値遺伝子（長さ <see cref="GenomeLayout.BitCount"/>）を取得します。</para>
    /// </summary>
    public ReadOnlySpan<bool> Bits { get; }

    /// <summary>
    /// Gets the categorical genes (length <see cref="GenomeLayout.CategoryCount"/>, each in <c>[0, count − 1]</c>).
    /// <para>カテゴリ遺伝子（長さ <see cref="GenomeLayout.CategoryCount"/>、各値は 0 からカテゴリ数 − 1）を取得します。</para>
    /// </summary>
    public ReadOnlySpan<int> Categories { get; }

    /// <summary>
    /// Gets all permutations concatenated in order (length <see cref="GenomeLayout.PermutationElementCount"/>).
    /// <para>すべての順列を順に連結したスパン（長さ <see cref="GenomeLayout.PermutationElementCount"/>）を取得します。</para>
    /// </summary>
    public ReadOnlySpan<int> Permutations => permutations;

    /// <summary>
    /// Gets a view of the specified permutation.
    /// <para>指定した順列のビューを取得します。</para>
    /// </summary>
    /// <param name="index">The zero-based permutation index. / 0 から始まる順列インデックス。</param>
    /// <returns>A read-only span containing a permutation of <c>0..length−1</c>. / <c>0..length−1</c> の置換を表す読み取り専用スパン。</returns>
    public ReadOnlySpan<int> Permutation(int index) => permutations.Slice(Layout.PermutationOffset(index), Layout.PermutationLengths[index]);

    /// <summary>
    /// Copies this genome view into an independent immutable <see cref="Genome"/>.
    /// <para>このゲノムビューの内容を独立した不変の <see cref="Genome"/> にコピーします。</para>
    /// </summary>
    /// <returns>A new <see cref="Genome"/> containing copied gene values. / 遺伝子値をコピーした新しい <see cref="Genome"/>。</returns>
    public Genome ToGenome() => new(Layout, Reals.ToArray(), Integers.ToArray(), Bits.ToArray(), Categories.ToArray(), permutations.ToArray());
}

/// <summary>
/// Represents an immutable owned copy of a single individual's genome.
/// <para>1個体分のゲノムの不変コピーを表します。</para>
/// </summary>
public sealed class Genome
{
    private readonly double[] reals;
    private readonly long[] integers;
    private readonly bool[] bits;
    private readonly int[] categories;
    private readonly int[] permutations;

    internal Genome(GenomeLayout layout, double[] reals, long[] integers, bool[] bits, int[] categories, int[] permutations)
    {
        Layout = layout;
        this.reals = reals;
        this.integers = integers;
        this.bits = bits;
        this.categories = categories;
        this.permutations = permutations;
    }

    /// <summary>
    /// Gets the layout describing this genome.
    /// <para>このゲノムの構成を取得します。</para>
    /// </summary>
    public GenomeLayout Layout { get; }

    /// <inheritdoc cref="GenomeView.Reals"/>
    public ReadOnlySpan<double> Reals => reals;

    /// <inheritdoc cref="GenomeView.Integers"/>
    public ReadOnlySpan<long> Integers => integers;

    /// <inheritdoc cref="GenomeView.Bits"/>
    public ReadOnlySpan<bool> Bits => bits;

    /// <inheritdoc cref="GenomeView.Categories"/>
    public ReadOnlySpan<int> Categories => categories;

    /// <inheritdoc cref="GenomeView.Permutations"/>
    public ReadOnlySpan<int> Permutations => permutations;

    /// <inheritdoc cref="GenomeView.Permutation(int)"/>
    public ReadOnlySpan<int> Permutation(int index) => permutations.AsSpan(Layout.PermutationOffset(index), Layout.PermutationLengths[index]);

    /// <summary>
    /// Returns a <see cref="GenomeView"/> referencing this genome.
    /// <para>このゲノムを参照する <see cref="GenomeView"/> を返します。</para>
    /// </summary>
    /// <returns>A <see cref="GenomeView"/> over this genome's gene arrays. / このゲノムの遺伝子配列を参照する <see cref="GenomeView"/>。</returns>
    public GenomeView AsView() => new(Layout, reals, integers, bits, categories, permutations);
}

/// <summary>
/// Provides a read-only batch view of multiple individuals' genomes stored column-major per section (genes × individuals).
/// <para>複数個体のゲノムを区画ごとの列優先形式（遺伝子数 × 個体数、1列 = 1個体）で参照する読み取り専用ビューを表します。</para>
/// </summary>
/// <remarks>
/// <para>Valid only until the next call to <see cref="GeneticAlgorithm.Tell"/>.</para>
/// <para>次の <see cref="GeneticAlgorithm.Tell"/> 呼び出しまで有効です。</para>
/// </remarks>
public readonly ref struct GenomeBatch
{
    private readonly ReadOnlySpan<double> reals;
    private readonly ReadOnlySpan<long> integers;
    private readonly ReadOnlySpan<bool> bits;
    private readonly ReadOnlySpan<int> categories;
    private readonly ReadOnlySpan<int> permutations;

    internal GenomeBatch(GenomeLayout layout, int count, ReadOnlySpan<double> reals, ReadOnlySpan<long> integers, ReadOnlySpan<bool> bits,
        ReadOnlySpan<int> categories, ReadOnlySpan<int> permutations)
    {
        Layout = layout;
        Count = count;
        this.reals = reals[..(layout.RealCount * count)];
        this.integers = integers[..(layout.IntegerCount * count)];
        this.bits = bits[..(layout.BitCount * count)];
        this.categories = categories[..(layout.CategoryCount * count)];
        this.permutations = permutations[..(layout.PermutationElementCount * count)];
    }

    /// <summary>
    /// Gets the layout shared by all genomes in this batch.
    /// <para>このバッチ内の全ゲノムに共通する構成を取得します。</para>
    /// </summary>
    public GenomeLayout Layout { get; }

    /// <summary>
    /// Gets the number of individuals in this batch.
    /// <para>このバッチに含まれる個体数を取得します。</para>
    /// </summary>
    public int Count { get; }

    /// <summary>
    /// Gets a view of the genome at the specified individual index.
    /// <para>指定したインデックスの個体のゲノムビューを取得します。</para>
    /// </summary>
    /// <param name="index">The zero-based individual index. / 0 から始まる個体インデックス。</param>
    /// <returns>The <see cref="GenomeView"/> for individual <paramref name="index"/>. / インデックス <paramref name="index"/> の個体の <see cref="GenomeView"/>。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is negative or greater than or equal to <see cref="Count"/>. / <paramref name="index"/> が 0 未満、または <see cref="Count"/> 以上の場合。</exception>
    public GenomeView this[int index]
    {
        get
        {
            ArgumentOutOfRangeException.ThrowIfNegative(index);
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, Count);
            GenomeLayout layout = Layout;
            return new GenomeView(layout,
                reals.Slice(index * layout.RealCount, layout.RealCount),
                integers.Slice(index * layout.IntegerCount, layout.IntegerCount),
                bits.Slice(index * layout.BitCount, layout.BitCount),
                categories.Slice(index * layout.CategoryCount, layout.CategoryCount),
                permutations.Slice(index * layout.PermutationElementCount, layout.PermutationElementCount));
        }
    }

    /// <summary>
    /// Gets the real-valued genes of all individuals as a matrix view of size <see cref="GenomeLayout.RealCount"/> × <see cref="Count"/>.
    /// <para>全個体の実数遺伝子を行列ビュー（<see cref="GenomeLayout.RealCount"/> × <see cref="Count"/>）として取得します。</para>
    /// </summary>
    public ReadOnlyMatrixSpan<double> Reals => new(reals, Layout.RealCount, Count);
}
