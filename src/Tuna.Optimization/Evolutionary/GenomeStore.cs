namespace Tuna.Optimization.Evolutionary;

// Column-major (genes × individuals) storage buffer per genome section.
internal sealed class GenomeStore
{
    private readonly GenomeLayout layout;

    public GenomeStore(GenomeLayout layout, int capacity)
    {
        this.layout = layout;
        Reals = new double[layout.RealCount * capacity];
        Integers = new long[layout.IntegerCount * capacity];
        Bits = new bool[layout.BitCount * capacity];
        Categories = new int[layout.CategoryCount * capacity];
        Permutations = new int[layout.PermutationElementCount * capacity];
    }

    public double[] Reals { get; }

    public long[] Integers { get; }

    public bool[] Bits { get; }

    public int[] Categories { get; }

    public int[] Permutations { get; }

    public Span<double> RealsOf(int column) => Reals.AsSpan(column * layout.RealCount, layout.RealCount);

    public Span<long> IntegersOf(int column) => Integers.AsSpan(column * layout.IntegerCount, layout.IntegerCount);

    public Span<bool> BitsOf(int column) => Bits.AsSpan(column * layout.BitCount, layout.BitCount);

    public Span<int> CategoriesOf(int column) => Categories.AsSpan(column * layout.CategoryCount, layout.CategoryCount);

    public Span<int> PermutationsOf(int column) => Permutations.AsSpan(column * layout.PermutationElementCount, layout.PermutationElementCount);

    public void CopyTo(int column, GenomeStore target, int targetColumn)
    {
        RealsOf(column).CopyTo(target.RealsOf(targetColumn));
        IntegersOf(column).CopyTo(target.IntegersOf(targetColumn));
        BitsOf(column).CopyTo(target.BitsOf(targetColumn));
        CategoriesOf(column).CopyTo(target.CategoriesOf(targetColumn));
        PermutationsOf(column).CopyTo(target.PermutationsOf(targetColumn));
    }

    public void CopyLeadingTo(int count, GenomeStore target, int targetColumn)
    {
        Reals.AsSpan(0, count * layout.RealCount).CopyTo(target.Reals.AsSpan(targetColumn * layout.RealCount));
        Integers.AsSpan(0, count * layout.IntegerCount).CopyTo(target.Integers.AsSpan(targetColumn * layout.IntegerCount));
        Bits.AsSpan(0, count * layout.BitCount).CopyTo(target.Bits.AsSpan(targetColumn * layout.BitCount));
        Categories.AsSpan(0, count * layout.CategoryCount).CopyTo(target.Categories.AsSpan(targetColumn * layout.CategoryCount));
        Permutations.AsSpan(0, count * layout.PermutationElementCount).CopyTo(target.Permutations.AsSpan(targetColumn * layout.PermutationElementCount));
    }

    public GenomeView View(int column) =>
        new(layout, RealsOf(column), IntegersOf(column), BitsOf(column), CategoriesOf(column), PermutationsOf(column));

    public GenomeBatch Batch(int count) => new(layout, count, Reals, Integers, Bits, Categories, Permutations);
}
