using CsCheck;
using Tuna.LinearAlgebra;
using static Tuna.Tests.LinearAlgebra.LinearAlgebraTestUtilities;

namespace Tuna.Tests.LinearAlgebra;

[Trait("LinearAlgebra", "SVD")]
[Collection(nameof(ParallelismCollection))]
public class SvdSolveTests
{
    [Fact]
    [Trait("Category", "Property")]
    public void SolvesLeastSquaresWithMinimumNormAndSatisfiesPenroseEquations()
    {
        var gen = Gen.Select(Gen.Int[1, 12], Gen.Int[1, 12], Gen.Int[0, 100000], Gen.Int[0, 100000]);
        gen.Sample((m, n, rankSeed, dataSeed) =>
        {
            int rank = rankSeed % (Math.Min(m, n) + 1);
            var sigma = new Matrix<double>(m, n);
            for (int i = 0; i < rank; i++)
            {
                sigma[i, i] = 1 + 0.25 * i;
            }

            var a = MultiplyNaive(MultiplyNaive(RandomOrthogonal(m, dataSeed), sigma), RandomOrthogonal(n, dataSeed + 1), transposeB: true);
            var b = Random(m, 3, dataSeed + 2);
            var svd = a.Svd();
            Assert.Equal(rank, svd.Rank);

            var inverse = svd.PseudoInverse();
            var aInverse = MultiplyNaive(a, inverse);
            var inverseA = MultiplyNaive(inverse, a);
            int dim = Math.Max(m, n);
            AssertRatio(Difference(a, MultiplyNaive(aInverse, a)), Norm(a), dim, "AA⁺A = A");
            AssertRatio(Difference(inverse, MultiplyNaive(inverseA, inverse)), Norm(inverse), dim, "A⁺AA⁺ = A⁺");
            AssertRatio(Difference(aInverse, aInverse.T.ToMatrix()), Norm(aInverse), dim, "AA⁺ is symmetric");
            AssertRatio(Difference(inverseA, inverseA.T.ToMatrix()), Norm(inverseA), dim, "A⁺A is symmetric");

            var x = svd.Solve(b);
            var atAx = MultiplyNaive(a, MultiplyNaive(a, x), transposeA: true);
            var atB = MultiplyNaive(a, b, transposeA: true);
            AssertRatio(Difference(atAx, atB), Norm(a) * Norm(b), dim, "Aᵀ(Ax − b) = 0");
            AssertRatio(Difference(x, MultiplyNaive(inverseA, x)), Norm(x), dim, "x belongs to the row space of A");
            AssertRatio(Difference(x, MultiplyNaive(inverse, b)), Norm(x), dim, "x = A⁺b");
        }, iter: 150, seed: "456789abcdef");
    }

    [Fact]
    public void UsesTheSameStrictRelativeCutoffForSolveAndPseudoInverse()
    {
        double small = 5 * Epsilon * 4;
        var a = new Matrix<double>(6, 4);
        a[0, 0] = 4;
        a[1, 1] = 2;
        a[2, 2] = small;
        var svd = a.Svd();
        var b = new Vector<double>([8, 6, 4 * small, 9, 7, -2]);
        Assert.Equal(2, svd.Rank);
        Assert.Equal(new double[] { 2, 3, 0, 0 }, svd.Solve(b).ToArray());
        Assert.Equal(new double[] { 2, 0, 0, 0 }, svd.Solve(b, tolerance: 0.5).ToArray());
        Assert.Equal(new double[] { 2, 3, 4, 0 }, svd.Solve(b, tolerance: 0).ToArray());
        Assert.Equal(new double[4], svd.Solve(b, tolerance: 1).ToArray());

        var matrix = Matrix<double>.FromColumnMajor(6, 2, [8, 6, 4 * small, 9, 7, -2, -4, 2, small, 0, 0, 0]);
        var destination = new Matrix<double>(4, 2);
        svd.Solve(matrix, destination, tolerance: 0.5);
        Assert.Equal(new double[] { 2, 0, 0, 0, -1, 0, 0, 0 }, destination.AsSpan().ToArray());
        Assert.Equal(0, svd.PseudoInverse()[2, 2]);
        Assert.Equal(0, svd.PseudoInverse(tolerance: 0.5)[1, 1]);
        Assert.Equal(1 / small, svd.PseudoInverse(tolerance: 0)[2, 2]);
    }

    [Fact]
    public void SolvesWithTheUpdatedRankAndFactorsAfterRecompute()
    {
        var svd = Matrix<double>.FromRowMajor(2, 3, [2, 0, 0, 0, 4, 0]).Svd();
        var b = new Vector<double>([6, 8]);
        Assert.Equal(new double[] { 3, 2, 0 }, svd.Solve(b).ToArray());

        svd.Recompute(Matrix<double>.FromRowMajor(2, 3, [0, 3, 0, 0, 0, 0]));
        Assert.Equal(1, svd.Rank);
        Assert.Equal(new double[] { 0, 2, 0 }, svd.Solve(b).ToArray());
        Assert.Equal(new double[,] { { 0, 0 }, { 1.0 / 3, 0 }, { 0, 0 } }, svd.PseudoInverse().ToArray());
    }

    [Fact]
    public void AcceptsOverlappingRightHandSidesAndWritesOnlyToDestinationView()
    {
        var svd = new Matrix<double>(new double[,] { { 2, 0 }, { 0, 5 }, { 0, 0 } }).Svd();
        double[] storage = [6, -99, 20, -99, 8];
        var b = new ReadOnlyVectorSpan<double>(storage, [3], [2]);
        var x = new VectorSpan<double>(storage.AsSpan(2), [2], [2]);
        var written = svd.Solve(b, x, tolerance: 0);
        Assert.Equal(new double[] { 3, 4 }, written.ToArray());
        Assert.Equal(new double[] { 6, -99, 3, -99, 4 }, storage);

        var matrix = new Matrix<double>(new double[,] { { 6, 10, -99 }, { 20, 30, -99 }, { 88, 99, -99 } });
        var matrixDestination = matrix[..2, ..2].T;
        var matrixWritten = svd.Solve(matrix[.., ..2], matrixDestination);
        Assert.Equal(new double[,] { { 3, 5 }, { 4, 6 } }, matrixWritten.ToArray());
        Assert.Equal(new double[] { 88, 99, -99 }, matrix.Row(2).ToArray());
        Assert.Equal(new double[] { -99, -99, -99 }, matrix.Column(2).ToArray());

        var inverseStorage = new Matrix<double>(4, 3);
        inverseStorage.AsSpan().Fill(-99);
        var inverseDestination = inverseStorage[1..4, 1..3].T;
        var inverseWritten = svd.PseudoInverse(inverseDestination, tolerance: 0.4);
        Assert.Equal(new double[,] { { 0, 0, 0 }, { 0, 0.2, 0 } }, inverseWritten.ToArray());
        Assert.Equal(new double[] { -99, -99, -99 }, inverseStorage.Row(0).ToArray());
        Assert.Equal(new double[] { -99, -99, -99, -99 }, inverseStorage.Column(0).ToArray());
    }

    [Fact]
    public void SupportsLargeWorkspacesAndStridedViews()
    {
        var a = Random(36, 20, 532);
        var expected = Random(20, 21, 533);
        var b = MultiplyNaive(a, expected);
        var stridedStorage = new double[40 * 21];
        Array.Fill(stridedStorage, -99);
        var destination = new MatrixSpan<double>(stridedStorage, [20, 21], [2, 40]);
        var svd = a.Svd();
        svd.Solve(b, destination);
        AssertRatio(Difference(expected, destination.ToMatrix()), Norm(expected), a.Rows, "x from strided destination");
        Assert.All(Enumerable.Range(0, 420), i => Assert.Equal(-99, stridedStorage[2 * i + 1]));
        AssertRatio(Difference(a, MultiplyNaive(MultiplyNaive(a, svd.PseudoInverse()), a)), Norm(a), a.Rows, "AA⁺A = A");
    }

    [Fact]
    public void SolvesLargeWideRankDeficientSystems()
    {
        const int m = 40, n = 60, rank = 32, rhsCount = 10;
        var u = RandomOrthogonal(m, 821);
        var v = RandomOrthogonal(n, 822);
        var sigma = new Matrix<double>(m, n);
        var reciprocal = new Matrix<double>(n, m);
        for (int i = 0; i < rank; i++)
        {
            sigma[i, i] = 1.0 + i / 32.0;
            reciprocal[i, i] = 1 / sigma[i, i];
        }

        var a = MultiplyNaive(MultiplyNaive(u, sigma), v, transposeB: true);
        var expectedInverse = MultiplyNaive(MultiplyNaive(v, reciprocal), u, transposeB: true);
        var rhs = Random(m, rhsCount, 823);
        var expected = MultiplyNaive(expectedInverse, rhs);
        var svd = a.Svd();
        Assert.Equal(rank, svd.Rank);

        var inverse = new Matrix<double>(n, m);
        svd.PseudoInverse(inverse);
        AssertRatio(Difference(expectedInverse, inverse), Norm(expectedInverse), n, "A⁺");

        var solutions = new Matrix<double>(n, rhsCount);
        svd.Solve(rhs, solutions);
        AssertRatio(Difference(expected, solutions), Norm(expected), n, "X = A⁺B");

        var solution = new Vector<double>(n);
        svd.Solve(rhs.Column(0), solution);
        var single = Matrix<double>.FromColumnMajor(n, 1, solution.AsSpan());
        var expectedSingle = expected[.., ..1].ToMatrix();
        AssertRatio(Difference(expectedSingle, single), Norm(expectedSingle), n, "x = A⁺b");
    }

    [Theory]
    [InlineData(8, 6, 3)]
    [InlineData(40, 60, 10)]
    public void DestinationOverloadsAllocateNothingAfterWarmup(int m, int n, int rhsCount)
    {
        var svd = Random(m, n, 912).Svd();
        var rhs = Random(m, rhsCount, 913);
        var vector = new Vector<double>(n);
        var matrix = new Matrix<double>(n, rhsCount);
        var inverse = new Matrix<double>(n, m);

        ParallelAllocationTests.AssertNoAllocation(() => svd.Solve(rhs.Column(0), vector));
        ParallelAllocationTests.AssertNoAllocation(() => svd.Solve(rhs, matrix));
        ParallelAllocationTests.AssertNoAllocation(() => svd.PseudoInverse(inverse));
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(3, 0)]
    [InlineData(0, 4)]
    [InlineData(4, 3)]
    public void PreservesEmptyShapesAndOverwritesDestinationsForZeroRank(int m, int n)
    {
        var svd = new Matrix<double>(m, n).Svd();
        var b = new Vector<double>(m);
        b.AsSpan().Fill(double.NaN);
        Assert.Equal(new double[n], svd.Solve(b).ToArray());

        var x = new Matrix<double>(n, 3);
        x.AsSpan().Fill(1);
        svd.Solve(b.Broadcast(3), x);
        Assert.Equal(new double[n * 3], x.AsSpan().ToArray());
        var empty = svd.Solve(new Matrix<double>(m, 0));
        Assert.Equal((n, 0), (empty.Rows, empty.Columns));

        var inverse = svd.PseudoInverse();
        Assert.Equal((n, m), (inverse.Rows, inverse.Columns));
        inverse.AsSpan().Fill(1);
        svd.PseudoInverse(inverse);
        Assert.Equal(new double[n * m], inverse.AsSpan().ToArray());
    }

    [Fact]
    public void HandlesSubnormalProjectionAndOverflowOfIntermediateCoefficients()
    {
        double tiny = double.Epsilon;
        var subnormal = new Matrix<double>(new double[,] { { 3 * tiny }, { 4 * tiny } }).Svd();
        Assert.Equal(0.12, subnormal.Solve(new Vector<double>([tiny, 0]))[0], 1e-15);

        double large = 0.9 * double.MaxValue;
        var tall = new Matrix<double>(new double[,] { { 1 }, { 1 } }).Svd();
        Assert.Equal(0.9, tall.Solve(new Vector<double>([large, large]))[0] / double.MaxValue, 1e-15);

        var wide = new Matrix<double>(new double[,] { { 3e-309, 3e-309 } }).Svd();
        var solution = wide.Solve(new Vector<double>([0.9]));
        var inverse = wide.PseudoInverse();
        for (int i = 0; i < 2; i++)
        {
            Assert.Equal(1.5, solution[i] / 1e308, 1e-14);
            Assert.Equal(5.0 / 3, inverse[i, 0] / 1e308, 1e-14);
        }

        var diagonal = Matrix<double>.FromDiagonal([1, tiny]).Svd();
        Assert.Equal(new double[] { 2, 1 }, diagonal.Solve(new Vector<double>([2, tiny]), tolerance: 0).ToArray());
        Assert.Equal(new double[] { 0, double.PositiveInfinity }, diagonal.Solve(new Vector<double>([0, 1]), tolerance: 0).ToArray());
        Assert.Equal(new double[,] { { 1, 0 }, { 0, double.PositiveInfinity } }, diagonal.PseudoInverse(tolerance: 0).ToArray());
    }

    [Fact]
    public void ComparesSubnormalSingularValuesBeforeRoundingTheCutoff()
    {
        double tiny = double.Epsilon;
        var svd = Matrix<double>.FromDiagonal([3 * tiny, 2 * tiny]).Svd();
        Assert.Equal(new double[] { 1, 1 }, svd.Solve(new Vector<double>([3 * tiny, 2 * tiny]), tolerance: 0.5).ToArray());
        Assert.Equal(double.PositiveInfinity, svd.PseudoInverse(tolerance: 0.5)[1, 1]);
    }

    [Fact]
    public void PreservesSmallSolutionComponentsAlongsideOverflowingOnes()
    {
        var svd = Matrix<double>.FromDiagonal([1, 0.5]).Svd();
        var b = Matrix<double>.FromColumnMajor(2, 3, [2, 4, double.Epsilon, double.MaxValue, 4, 6]);
        svd.Solve(b, b);
        Assert.Equal(new double[] { 2, 8, double.Epsilon, double.PositiveInfinity, 4, 12 }, b.AsSpan().ToArray());
    }

    [Fact]
    public void PseudoInverseHandlesCancellationBetweenTermsThatOverflow()
    {
        var a = new Matrix<double>(new double[,] { { 1e-310, 1e-310 }, { 0, 1e-310 } });
        var inverse = a.Svd().PseudoInverse();
        Assert.Equal(double.PositiveInfinity, inverse[0, 0]);
        Assert.Equal(double.NegativeInfinity, inverse[0, 1]);
        Assert.InRange(Math.Abs(inverse[1, 0]), 0, 1e300);
        Assert.Equal(double.PositiveInfinity, inverse[1, 1]);
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void RejectsNonFiniteSingularValuesAndPropagatesNonFiniteRightHandSides(double value)
    {
        var svd = new Matrix<double>(new double[,] { { value } }).Svd();
        Assert.Throws<InvalidOperationException>(() => svd.Solve(new Vector<double>([1])));
        Assert.Throws<InvalidOperationException>(() => svd.PseudoInverse());

        var identity = Matrix<double>.FromDiagonal([1]).Svd();
        Assert.Equal(value, identity.Solve(new Vector<double>([value]))[0]);
    }

    [Theory]
    [InlineData(-double.Epsilon)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void RejectsInvalidTolerances(double tolerance)
    {
        var svd = Matrix<double>.FromDiagonal([1]).Svd();
        Assert.Throws<ArgumentOutOfRangeException>("tolerance", () => svd.Solve(new Vector<double>([1]), tolerance));
        Assert.Throws<ArgumentOutOfRangeException>("tolerance", () => svd.PseudoInverse(tolerance));
    }

    [Fact]
    public void RejectsMissingVectorsInvalidShapesAndDestinationsThatOverlapFactors()
    {
        var a = Matrix<double>.FromDiagonal([2, 1]);
        var valuesOnly = a.Svd(computeVectors: false);
        Assert.Throws<InvalidOperationException>(() => valuesOnly.Solve(a));
        Assert.Throws<InvalidOperationException>(() => valuesOnly.PseudoInverse());

        var svd = a.Svd();
        var b = new Vector<double>([1, 2]);
        Assert.Throws<ArgumentException>("b", () => svd.Solve(new Vector<double>(1)));
        Assert.Throws<ArgumentException>("b", () => svd.Solve(new Matrix<double>(3, 2)));
        Assert.Throws<ArgumentException>("destination", () => svd.Solve(b, new Vector<double>(3)));
        Assert.Throws<ArgumentException>("destination", () => svd.Solve(a, new Matrix<double>(2, 1)));
        Assert.Throws<ArgumentException>("destination", () => svd.PseudoInverse(new Matrix<double>(2, 1)));
        Assert.Throws<ArgumentException>("destination", () => svd.Solve(a, svd.U!));
        Assert.Throws<ArgumentException>("destination", () => svd.PseudoInverse(svd.V!));
        Assert.Throws<ArgumentException>("destination", () => svd.Solve(b, new VectorSpan<double>(svd.Values)));
        Assert.Equal(new double[] { 2, 1 }, svd.Values);
    }
}
