using System.Numerics;
using System.Runtime.InteropServices;
using Tuna.LinearAlgebra;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>値型の並列作業でも、区間の端・入力と同じ出力先・同時呼び出しの結果を保つ</summary>
[Trait("LinearAlgebra", "Parallelism")]
[Collection(nameof(ParallelismCollection))]
public class Level1UpdateTests
{
    private delegate void Update<T>(Span<T> x, Span<T> y);

    private static readonly Update<float>[] SingleUpdates =
    [
        (x, y) => Blas.Axpy(0.37f, x, y),
        (x, y) => Blas.Scale(-1.7f, x),
        (x, y) => Blas.Rotate(x, y, 0.8f, 0.6f),
        (x, y) => Blas.Swap(x, y),
        (x, y) => Blas.Copy(x, y),
    ];

    private static readonly Update<double>[] DoubleUpdates =
    [
        (x, y) => Blas.Axpy(0.37, x, y),
        (x, y) => Blas.Scale(-1.7, x),
        (x, y) => Blas.Rotate(x, y, 0.8, 0.6),
        (x, y) => Blas.Swap(x, y),
        (x, y) => Blas.Copy(x, y),
    ];

    private static readonly Update<Complex>[] ComplexUpdates =
    [
        (x, y) => Blas.Axpy(new Complex(0.37, -0.2), x, y),
        (x, y) => Blas.Scale(new Complex(-1.7, 0.3), x),
        (x, y) => Blas.Scale(-1.7, x),
        (x, y) => Blas.Rotate(x, y, 0.8, 0.6),
        (x, y) => Blas.Swap(x, y),
        (x, y) => Blas.Copy(x, y),
    ];

    [Theory]
    [InlineData(0, false)]
    [InlineData(1, false)]
    [InlineData(32_769, false)]
    [InlineData(65_537, false)]
    [InlineData(300_007, false)]
    [InlineData(300_007, true)]
    public void UpdatesMatchSingleThreadBitwise(int length, bool alias)
    {
        var x = Enumerable.Range(0, length + 8).Select(i => (i % 31 - 15) * 0.0123).ToArray();
        var y = Enumerable.Range(0, length + 8).Select(i => (i % 23 - 11) * 0.0017).ToArray();
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            CheckUpdates(DoubleUpdates, x, y, length, alias);
            CheckUpdates(SingleUpdates, Array.ConvertAll(x, v => (float)v), Array.ConvertAll(y, v => (float)v), length, alias);
            CheckUpdates(ComplexUpdates,
                x.Zip(y, (r, i) => new Complex(r, i)).ToArray(),
                y.Zip(x, (r, i) => new Complex(r, -i)).ToArray(), length, alias);
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    private static void CheckUpdates<T>(Update<T>[] updates, T[] x, T[] y, int length, bool alias)
        where T : unmanaged
    {
        foreach (var update in updates)
        {
            T[] expectedX = x.ToArray();
            T[] expectedY = alias ? expectedX : y.ToArray();
            LinearAlgebraSettings.MaxDegreeOfParallelism = 1;
            update(expectedX.AsSpan(3, length), expectedY.AsSpan(3, length));
            foreach (int threads in new[] { 2, 3, 4 })
            {
                T[] actualX = x.ToArray();
                T[] actualY = alias ? actualX : y.ToArray();
                LinearAlgebraSettings.MaxDegreeOfParallelism = threads;
                update(actualX.AsSpan(3, length), actualY.AsSpan(3, length));
                AssertSameBits(expectedX, actualX);
                AssertSameBits(expectedY, actualY);
            }
        }
    }

    [Fact]
    public void ConcurrentUpdatesUseSeparateState()
    {
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            CheckConcurrentUpdates(SingleUpdates);
            CheckConcurrentUpdates(DoubleUpdates);
            CheckConcurrentUpdates(ComplexUpdates);
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    private static void CheckConcurrentUpdates<T>(Update<T>[] updates)
        where T : unmanaged, INumberBase<T>
    {
        const int length = 300_007, callers = 4;
        foreach (var update in updates)
        {
            var xs = new T[callers][];
            var ys = new T[callers][];
            var expectedXs = new T[callers][];
            var expectedYs = new T[callers][];
            LinearAlgebraSettings.MaxDegreeOfParallelism = 1;
            for (int i = 0; i < callers; i++)
            {
                xs[i] = new T[length];
                ys[i] = new T[length];
                Array.Fill(xs[i], T.CreateChecked(0.25 * (i + 1)));
                Array.Fill(ys[i], T.CreateChecked(-0.5 * (i + 1)));
                expectedXs[i] = xs[i].ToArray();
                expectedYs[i] = ys[i].ToArray();
                update(expectedXs[i], expectedYs[i]);
            }

            LinearAlgebraSettings.MaxDegreeOfParallelism = 4;
            Parallel.For(0, callers, new ParallelOptions { MaxDegreeOfParallelism = callers }, i => update(xs[i], ys[i]));
            for (int i = 0; i < callers; i++)
            {
                AssertSameBits(expectedXs[i], xs[i]);
                AssertSameBits(expectedYs[i], ys[i]);
            }
        }
    }

    private static void AssertSameBits<T>(T[] expected, T[] actual)
        where T : unmanaged =>
        Assert.True(MemoryMarshal.AsBytes(expected.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(actual.AsSpan())));
}
