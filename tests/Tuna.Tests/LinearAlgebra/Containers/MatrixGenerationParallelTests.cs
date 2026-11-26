using Tuna.LinearAlgebra;
using static Tuna.Tests.LinearAlgebra.LinearAlgebraTestUtilities;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>並列の経路: スレッド数によらずビット単位で同じ結果、各要素を1回だけ求める、演算の例外をそのまま伝える</summary>
[Trait("LinearAlgebra", "Parallelism")]
[Collection(nameof(ParallelismCollection))]
public class MatrixGenerationParallelTests
{
    private static T WithThreads<T>(int threads, Func<T> compute)
    {
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = threads;
            return compute();
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    private static int Threads => Math.Max(4, Environment.ProcessorCount);

    [Fact]
    public void ResultsDoNotDependOnTheNumberOfThreads()
    {
        var x = Random(6, 450, 31);
        var y = Random(6, 380, 32);
        var op = new XMath.Operators.ScaledSquaredDistance<double>([0.5, 1, 1.5, 2, 2.5, 3]);
        Matrix<double>[] Compute() =>
        [
            XMath.Pairwise(x, y, op),
            XMath.PairwiseSymmetric(x, op),
            XMath.PairwiseSymmetric<XMath.Operators.Distance<double>>(x),
            Matrix<double>.GenerateSymmetric(900, 5, new MatrixGenerationTests.Toeplitz([4, -1, 0.5, -0.25, 0.125, -0.0625])),
            Matrix<double>.GenerateBand<MatrixGenerationTests.Hilbert>(700, 1200, 3, 40),
            Matrix<double>.Generate<MatrixGenerationTests.Hilbert>(300, 500),
        ];

        var sequential = WithThreads(1, Compute);
        var parallel = WithThreads(Threads, Compute);
        for (int k = 0; k < sequential.Length; k++)
        {
            Assert.Equal(sequential[k].AsSpan().ToArray(), parallel[k].AsSpan().ToArray());
        }

        Assert.Equal(sequential[1].ToArray(), XMath.Transpose(sequential[1]).ToArray());
    }

    [Fact]
    public void EachElementOfTheBandIsEvaluatedOnce()
    {
        int rows = 1500, columns = 1300, lower = 7, upper = 2;
        var calls = new int[rows * columns];
        var m = WithThreads(Threads, () =>
            Matrix<double>.GenerateBand(rows, columns, lower, upper, new MatrixGenerationTests.BandChecked(rows, lower, upper, calls)));
        for (int j = 0; j < columns; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                bool inside = j - i <= upper && i - j <= lower;
                Assert.Equal(inside ? 1 : 0, calls[i + j * rows]);
                Assert.Equal(inside ? MatrixGenerationTests.BandChecked.Value(i, j) : 0, m[i, j]);
            }
        }

        int n = 1200, bandwidth = 4;
        var symmetricCalls = new int[n * n];
        WithThreads(Threads, () => Matrix<double>.GenerateSymmetric(n, bandwidth, new MatrixGenerationTests.BandChecked(n, bandwidth, 0, symmetricCalls)));
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                Assert.Equal(i >= j && i - j <= bandwidth ? 1 : 0, symmetricCalls[i + j * n]);
            }
        }
    }

    [Fact]
    public void ExceptionsFromTheOperatorPropagateUnchangedFromWorkerThreads()
    {
        var e = Assert.Throws<InvalidOperationException>(() =>
            WithThreads(Threads, () => Matrix<double>.Generate(600, 600, new MatrixGenerationTests.ThrowsAt(500, 400))));
        Assert.Equal("boom", e.Message);
        var expected = new InvalidOperationException("lambda");
        var failure = Assert.Throws<InvalidOperationException>(() =>
            WithThreads(Threads, () => Matrix<double>.Generate(600, 600, (i, j) => i == 500 && j == 400 ? throw expected : 0.0)));
        Assert.Same(expected, failure);
        var x = Random(2, 500, 3);
        x[0, 450] = x[0, 10];
        Assert.Throws<InvalidOperationException>(() =>
            WithThreads(Threads, () => XMath.PairwiseSymmetric<MatrixGenerationTests.ThrowingPair>(x)));
    }
}
