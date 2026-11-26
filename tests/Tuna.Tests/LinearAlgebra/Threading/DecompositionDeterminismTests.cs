using System.Numerics;
using System.Reflection;
using System.Runtime.InteropServices;
using Tuna.LinearAlgebra;
using static Tuna.Tests.LinearAlgebra.LinearAlgebraTestUtilities;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>分解の全出力は、並列数や作業スレッドを確保できるかどうかによらず同じビットになる</summary>
[Trait("LinearAlgebra", "Parallelism")]
[Collection(nameof(ParallelismCollection))]
public class DecompositionDeterminismTests
{
    private delegate void SolveRight(int m, int k, ReadOnlySpan<double> l, int ldl, Span<double> b, int ldb);
    private delegate void Update(int m, int k, ReadOnlySpan<double> a, int lda, Span<double> c, int ldc);
    private delegate void SymmetricMultiply(int n, double alpha, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> x, Span<double> y);

    /// <summary>縮約の末尾や小さなパネルだけでは通らない、内部カーネルの複数区間と端数も検査する</summary>
    [Theory]
    [InlineData(257, 33)]
    [InlineData(601, 257)]
    public void PartialSumsAndCholeskyBlocksAreIdentical(int m, int k)
    {
        var solve = Kernel<SolveRight>("CholeskyKernels", "SolveTransposedRight");
        var update = Kernel<Update>("CholeskyKernels", "UpdateTrailing");
        var multiply = Kernel<SymmetricMultiply>("MatrixVector", "SymmetricMultiply");
        var l = Random(k, k, 801);
        for (int j = 0; j < k; j++)
        {
            l[j, j] += k;
        }

        var b = Random(m, k, 802);
        var a = RandomSymmetric(m, 803);
        double[] x = Enumerable.Range(0, m).Select(i => Math.Sin(i)).ToArray();
        AssertDegrees(() =>
        {
            double[] solved = b.AsSpan().ToArray();
            double[] updated = a.AsSpan().ToArray();
            var product = new double[m];
            solve(m, k, l.AsSpan(), k, solved, m);
            update(m, k, b.AsSpan(), m, updated, m);
            multiply(m, -0.75, a.AsSpan(), m, x, product);
            return [Bits<double>(solved), Bits<double>(updated), Bits<double>(product)];
        });
    }

    [Theory]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(127)]
    [InlineData(128)]
    [InlineData(129)]
    [InlineData(257)]
    [InlineData(601)]
    [InlineData(899)]
    [InlineData(900)]
    [InlineData(1100)]
    public void CholeskyOutputsAreIdentical(int n)
    {
        var a = RandomPositiveDefinite(n, 810 + n);
        var b = Random(n, 7, 811 + n);
        AssertDegrees(() =>
        {
            var c = a.Cholesky();
            return [Bits<double>(c.L.AsSpan()), Bits<double>(c.Inverse().AsSpan()),
                Bits<double>(c.Solve(b).AsSpan()), Bits<double>([c.LogDeterminant])];
        });
    }

    [Theory]
    [InlineData(130)]
    [InlineData(257)]
    [InlineData(600)]
    [InlineData(1100)]
    public void EigenAndSvdOutputsAreIdentical(int n)
    {
        var symmetric = RandomSymmetric(n, 820 + n);
        var general = Random(n, n, 821 + n);
        var tall = Random(n + n / 2, n, 822 + n);
        AssertDegrees(() =>
        {
            var s = symmetric.SymmetricEigen();
            var e = general.Eigen();
            var v = tall.Svd();
            return [Bits<double>(s.Values), Bits<double>(s.Vectors!.AsSpan()),
                Bits<double>(SymmetricEigenDecomposition.Compute(symmetric, computeVectors: false).Values),
                Bits<Complex>(e.Values), Bits<double>(e.Vectors!.AsSpan()),
                Bits<Complex>(EigenDecomposition.Compute(general, computeVectors: false).Values),
                Bits<double>(v.Values), Bits<double>(v.U!.AsSpan()), Bits<double>(v.V!.AsSpan()),
                Bits<double>(SingularValueDecomposition.Compute(tall, computeVectors: false).Values)];
        });
    }

    [Theory]
    [InlineData(257, 257)]
    [InlineData(257, 601)]
    public void SquareAndWideSvdOutputsAreIdentical(int m, int n)
    {
        var a = Random(m, n, 830 + m + n);
        AssertDegrees(() =>
        {
            var v = a.Svd();
            return [Bits<double>(v.Values), Bits<double>(v.U!.AsSpan()), Bits<double>(v.V!.AsSpan()),
                Bits<double>(SingularValueDecomposition.Compute(a, computeVectors: false).Values)];
        });
    }

    [Fact]
    public void ConcurrentDecompositionsAreIdentical()
    {
        const int n = 300;
        var a = RandomPositiveDefinite(n, 840);
        var g = Random(n, n, 841);
        Func<byte[][]> run = () =>
        {
            var c = a.Cholesky();
            var s = a.SymmetricEigen();
            var e = g.Eigen();
            var v = g.Svd();
            return [Bits<double>(c.L.AsSpan()), Bits<double>(s.Values), Bits<double>(s.Vectors!.AsSpan()),
                Bits<Complex>(e.Values), Bits<double>(e.Vectors!.AsSpan()),
                Bits<double>(v.Values), Bits<double>(v.U!.AsSpan()), Bits<double>(v.V!.AsSpan())];
        };
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = 1;
            var expected = run();
            LinearAlgebraSettings.MaxDegreeOfParallelism = 4;
            Parallel.For(0, 8, new ParallelOptions { MaxDegreeOfParallelism = 4 }, _ => AssertBits(expected, run()));
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    private static byte[] Bits<T>(ReadOnlySpan<T> values)
        where T : unmanaged => MemoryMarshal.AsBytes(values).ToArray();

    private static T Kernel<T>(string type, string method)
        where T : Delegate => typeof(Matrix<double>).Assembly.GetType($"Tuna.LinearAlgebra.{type}")!
            .GetMethod(method, BindingFlags.Static | BindingFlags.NonPublic)!.CreateDelegate<T>();

    private static void AssertDegrees(Func<byte[][]> run)
    {
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = 1;
            var expected = run();
            for (int degree = 2; degree <= 4; degree++)
            {
                LinearAlgebraSettings.MaxDegreeOfParallelism = degree;
                AssertBits(expected, run());
            }
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    private static void AssertBits(byte[][] expected, byte[][] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(expected[i].AsSpan().SequenceEqual(actual[i]), $"Output {i} differs at degree {LinearAlgebraSettings.MaxDegreeOfParallelism}.");
        }
    }
}
