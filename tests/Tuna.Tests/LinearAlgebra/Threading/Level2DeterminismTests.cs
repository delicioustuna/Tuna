using System.Numerics;
using System.Runtime.InteropServices;
using Tuna.LinearAlgebra;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>行列とベクトルの演算は、最大並列数を変えても値のビット表現を保つ</summary>
public abstract class Level2DeterminismTests<T> where T : unmanaged, INumberBase<T>
{
    protected abstract void Gemv(BlasTranspose trans, int m, int n, T alpha, ReadOnlySpan<T> a, int lda, ReadOnlySpan<T> x, T beta, Span<T> y);
    protected abstract void Symv(BlasTriangle uplo, int n, T alpha, ReadOnlySpan<T> a, int lda, ReadOnlySpan<T> x, T beta, Span<T> y, bool hermitian);
    protected abstract void Ger(int m, int n, T alpha, ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> a, int lda, bool conjugate);
    protected abstract void Triangular(bool solve, BlasTriangle uplo, BlasTranspose trans, BlasDiagonal diag, int n, ReadOnlySpan<T> a, int lda, Span<T> x);
    protected abstract void ScaleMatrix(int m, int n, T beta, Span<T> a, int lda);

    // 小さい行列、分割閾値付近、列の部分和、行の分割と端数を転置ビューおよび destination オーバーロードでも比較する
    [Theory]
    [InlineData(127, 127)]
    [InlineData(129, 257)]
    [InlineData(257, 257)]
    [InlineData(1000, 1000)]
    [InlineData(997, 1009)]
    [InlineData(301, 4099)]
    [InlineData(1000, 64)]
    [InlineData(2000, 64)]
    [InlineData(2047, 64)]
    [InlineData(2047, 257)]
    [InlineData(2048, 257)]
    [InlineData(2049, 257)]
    [InlineData(8000, 64)]
    [InlineData(30_000, 16)]
    [InlineData(2047, 1)]
    [InlineData(2048, 1)]
    [InlineData(2049, 1)]
    [InlineData(2047, 3)]
    [InlineData(2048, 3)]
    [InlineData(2049, 3)]
    [InlineData(2047, 5)]
    [InlineData(2048, 5)]
    [InlineData(2049, 5)]
    [InlineData(1023, 64)]
    [InlineData(1023, 511)]
    [InlineData(1023, 512)]
    [InlineData(1023, 513)]
    [InlineData(1024, 64)]
    [InlineData(1024, 511)]
    [InlineData(1024, 512)]
    [InlineData(1024, 513)]
    [InlineData(1025, 64)]
    [InlineData(1025, 511)]
    [InlineData(1025, 512)]
    [InlineData(1025, 513)]
    [InlineData(4095, 64)]
    [InlineData(4095, 511)]
    [InlineData(4095, 512)]
    [InlineData(4095, 513)]
    [InlineData(4096, 64)]
    [InlineData(4096, 511)]
    [InlineData(4096, 512)]
    [InlineData(4096, 513)]
    [InlineData(4097, 64)]
    [InlineData(4097, 511)]
    [InlineData(4097, 512)]
    [InlineData(4097, 513)]
    [InlineData(65_539, 17)]
    public void MatrixVectorProductsHaveIdenticalBits(int m, int n)
    {
        var a = new Matrix<T>(m, n);
        Values(a.AsSpan(), 1200 + m + n);
        var x = new Tuna.LinearAlgebra.Vector<T>(n);
        var xt = new Tuna.LinearAlgebra.Vector<T>(m);
        Values(x.AsSpan(), 1300 + n);
        Values(xt.AsSpan(), 1400 + m);

        AcrossThreadCounts(() =>
        {
            var normal = XMath.Multiply<T>(a, x).ToArray();
            var transposed = XMath.Multiply<T>(a.T, xt).ToArray();
            var destination = new Tuna.LinearAlgebra.Vector<T>(m);
            var destinationT = new Tuna.LinearAlgebra.Vector<T>(n);
            XMath.Multiply<T>(a, x, destination);
            XMath.Multiply<T>(a.T, xt, destinationT);
            SameBits(normal, destination.AsSpan());
            SameBits(transposed, destinationT.AsSpan());
            return [normal, transposed];
        });
    }

    /// <summary>行が多く列が少ない場合も、行の端数や列の4要素の端数は並列数によらない</summary>
    [Theory]
    [InlineData(-1, 1)]
    [InlineData(0, 1)]
    [InlineData(1, 1)]
    [InlineData(-1, 3)]
    [InlineData(0, 3)]
    [InlineData(1, 3)]
    [InlineData(-1, 5)]
    [InlineData(0, 5)]
    [InlineData(1, 5)]
    public void NarrowProductsAtLargeRowCountsHaveIdenticalBits(int offset, int columns)
    {
        int rows = 256 * 1024 / Marshal.SizeOf<T>() + offset;
        var a = new Matrix<T>(rows, columns);
        var x = new Tuna.LinearAlgebra.Vector<T>(columns);
        var xt = new Tuna.LinearAlgebra.Vector<T>(rows);
        Values(a.AsSpan(), 2900 + rows + columns);
        Values(x.AsSpan(), 3000 + columns);
        Values(xt.AsSpan(), 3100 + rows);
        AcrossThreadCounts(() => [XMath.Multiply<T>(a, x).ToArray(), XMath.Multiply<T>(a.T, xt).ToArray()]);
    }

    /// <summary>作業スレッドが多くても、最後の短い行の範囲だけで複素数の SIMD 経路を変えない</summary>
    [Fact]
    public void RowProductsHaveIdenticalBitsWithManyWorkers()
    {
        var a = new Matrix<T>(2050, 1024);
        var x = new Tuna.LinearAlgebra.Vector<T>(a.Columns);
        Values(a.AsSpan(), 3200);
        Values(x.AsSpan(), 3201);
        AcrossThreadCounts(() => [XMath.Multiply<T>(a, x).ToArray()], 2, 3, 4, 8, 16, 17, 31, 32, 33);
    }

    /// <summary>係数・元の出力・列の間隔・共役転置も、並列数によらず同じ順に計算する</summary>
    [Theory]
    [InlineData(31, 29)]
    [InlineData(259, 257)]
    [InlineData(1003, 1013)]
    [InlineData(65_539, 9)]
    public void GemvWithScaledOutputAndPaddedColumnsHasIdenticalBits(int m, int n)
    {
        int lda = m + 3;
        var a = Values(lda * n, 1500 + m + n);
        var x = Values(Math.Max(m, n), 1600 + m);
        var initial = Values(Math.Max(m, n) + 4, 1700 + n);
        AcrossThreadCounts(() =>
        {
            var results = new List<T[]>();
            foreach (var trans in Enum.GetValues<BlasTranspose>())
            {
                int leny = trans == BlasTranspose.None ? m : n;
                var y = (T[])initial.Clone();
                Gemv(trans, m, n, Number(.7, -.3), a, lda, x, Number(-.375, .125), y.AsSpan(1, leny));
                SameBits(initial.AsSpan(0, 1), y.AsSpan(0, 1));
                SameBits(initial.AsSpan(leny + 1), y.AsSpan(leny + 1));
                results.Add(y);
            }

            return results.ToArray();
        });
    }

    /// <summary>他の並列区間の中から呼んで逐次に実行される場合も、分担と合計する順序を保つ</summary>
    [Fact]
    public void ProductsHaveIdenticalBitsInsideAnotherParallelCall()
    {
        var a = new Matrix<T>(1003, 997);
        Values(a.AsSpan(), 2600);
        var x = new Tuna.LinearAlgebra.Vector<T>(a.Columns);
        var xt = new Tuna.LinearAlgebra.Vector<T>(a.Rows);
        Values(x.AsSpan(), 2601);
        Values(xt.AsSpan(), 2602);
        var type = typeof(Matrix<T>).Assembly.GetType("Tuna.LinearAlgebra.Parallelism")!;
        var method = type.GetMethod("For", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic,
            [typeof(int), typeof(int), typeof(Action<int>)])!;
        var run = method.CreateDelegate<Action<int, int, Action<int>>>();
        T[][] Compute() => [XMath.Multiply<T>(a, x).ToArray(), XMath.Multiply<T>(a.T, xt).ToArray()];

        AcrossThreadCounts(() =>
        {
            var expected = Compute();
            var nested = new T[2][][];
            run(2, 2, index => nested[index] = Compute());
            foreach (var result in nested)
            {
                SameBits(expected[0], result[0]);
                SameBits(expected[1], result[1]);
            }

            return expected;
        });
    }

    /// <summary>積を省略した GEMM の行列の係数倍も、列の境界で SIMD の端数の扱いを変えない</summary>
    [Theory]
    [InlineData(127, 127)]
    [InlineData(259, 257)]
    [InlineData(1003, 1013)]
    public void MatrixScalingHasIdenticalBits(int m, int n)
    {
        var dense = Values(m * n, 2700 + m + n);
        var padded = Values((m + 3) * n, 2800 + m + n);
        AcrossThreadCounts(() =>
        {
            var a = (T[])dense.Clone();
            var b = (T[])padded.Clone();
            ScaleMatrix(m, n, Number(.7, -.3), a, m);
            ScaleMatrix(m, n, Number(.7, -.3), b, m + 3);
            for (int j = 0; j < n; j++)
            {
                SameBits(padded.AsSpan(j * (m + 3) + m, 3), b.AsSpan(j * (m + 3) + m, 3));
            }

            return [a, b];
        });
    }

    /// <summary>三角形の分割と2列ずつ処理する端数を、上下の対称行列とエルミート行列で比べる</summary>
    [Theory]
    [InlineData(127)]
    [InlineData(257)]
    [InlineData(513)]
    [InlineData(1000)]
    [InlineData(1003)]
    public void SymmetricProductsHaveIdenticalBits(int n)
    {
        int lda = n + 3;
        var a = Values(lda * n, 1800 + n);
        var x = Values(n, 1900 + n);
        var initial = Values(n, 2000 + n);
        AcrossThreadCounts(() =>
        {
            var results = new List<T[]>();
            foreach (var uplo in Enum.GetValues<BlasTriangle>())
            {
                foreach (bool hermitian in typeof(T) == typeof(Complex) ? new[] { false, true } : new[] { false })
                {
                    var y = (T[])initial.Clone();
                    Symv(uplo, n, Number(.7, -.3), a, lda, x, Number(-.375, .125), y, hermitian);
                    results.Add(y);
                }
            }

            return results.ToArray();
        });
    }

    /// <summary>階数1の更新は列ごとの演算と列の余りを、並列数を変えても保つ</summary>
    [Theory]
    [InlineData(31, 29)]
    [InlineData(259, 257)]
    [InlineData(1003, 1013)]
    public void RankOneUpdatesHaveIdenticalBits(int m, int n)
    {
        int lda = m + 3;
        var initial = Values(lda * n, 2100 + m + n);
        var x = Values(m, 2200 + m);
        var y = Values(n, 2300 + n);
        AcrossThreadCounts(() =>
        {
            var results = new List<T[]>();
            foreach (bool conjugate in typeof(T) == typeof(Complex) ? new[] { false, true } : new[] { false })
            {
                var a = (T[])initial.Clone();
                Ger(m, n, Number(.7, -.3), x, y, a, lda, conjugate);
                for (int j = 0; j < n; j++)
                {
                    SameBits(initial.AsSpan(j * lda + m, 3), a.AsSpan(j * lda + m, 3));
                }

                results.Add(a);
            }

            return results.ToArray();
        });
    }

    /// <summary>三角行列の積と求解の更新は、ブロックの端数・転置・単位対角でも同じビットになる</summary>
    [Theory]
    [InlineData(31)]
    [InlineData(65)]
    [InlineData(1003)]
    public void TriangularProductsAndSolvesHaveIdenticalBits(int n)
    {
        int lda = n + 3;
        var a = Values(lda * n, 2400 + n);
        for (int i = 0; i < a.Length; i++)
        {
            a[i] /= Number(n * 1024.0);
        }

        for (int i = 0; i < n; i++)
        {
            a[i + i * lda] = Number(2.3, .1);
        }

        var initial = Values(n, 2500 + n);
        AcrossThreadCounts(() =>
        {
            var results = new List<T[]>();
            foreach (bool solve in new[] { false, true })
            {
                foreach (var uplo in Enum.GetValues<BlasTriangle>())
                {
                    foreach (var trans in Enum.GetValues<BlasTranspose>())
                    {
                        foreach (var diag in Enum.GetValues<BlasDiagonal>())
                        {
                            var x = (T[])initial.Clone();
                            Triangular(solve, uplo, trans, diag, n, a, lda, x);
                            results.Add(x);
                        }
                    }
                }
            }

            return results.ToArray();
        });
    }

    private static T Number(double real, double imaginary = 0) =>
        typeof(T) == typeof(Complex) ? (T)(object)new Complex(real, imaginary) : T.CreateChecked(real);

    private static T[] Values(int count, int seed)
    {
        var result = new T[count];
        Values(result, seed);
        return result;
    }

    private static void Values(Span<T> values, int seed)
    {
        var random = new Random(seed);
        for (int i = 0; i < values.Length; i++)
        {
            double real = Math.ScaleB(2 * random.NextDouble() - 1, random.Next(-8, 9));
            double imaginary = Math.ScaleB(2 * random.NextDouble() - 1, random.Next(-8, 9));
            values[i] = Number(real, imaginary);
        }
    }

    private static void AcrossThreadCounts(Func<T[][]> compute, params int[] degrees)
    {
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = 1;
            var expected = compute();
            foreach (var result in expected)
            {
                Assert.All(result, value => Assert.True(T.IsFinite(value)));
            }

            foreach (int degree in degrees.Length == 0 ? new[] { 2, 3, 4, 8, 3 } : degrees)
            {
                LinearAlgebraSettings.MaxDegreeOfParallelism = degree;
                var actual = compute();
                Assert.Equal(expected.Length, actual.Length);
                for (int i = 0; i < expected.Length; i++)
                {
                    SameBits(expected[i], actual[i]);
                }
            }
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    private static void SameBits(ReadOnlySpan<T> expected, ReadOnlySpan<T> actual) =>
        Assert.True(MemoryMarshal.AsBytes(expected).SequenceEqual(MemoryMarshal.AsBytes(actual)), "並列数によって値のビット表現が変わった");
}

[Collection(nameof(ParallelismCollection))]
public sealed class DoubleLevel2DeterminismTests : Level2DeterminismTests<double>
{
    protected override void ScaleMatrix(int m, int n, double beta, Span<double> a, int lda) =>
        Blas.Gemm(BlasTranspose.None, BlasTranspose.None, m, n, 0, 0.0, [], m, [], 1, beta, a, lda);

    protected override void Gemv(BlasTranspose trans, int m, int n, double alpha, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> x, double beta, Span<double> y) =>
        Blas.Gemv(trans, m, n, alpha, a, lda, x, beta, y);

    protected override void Symv(BlasTriangle uplo, int n, double alpha, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> x, double beta, Span<double> y, bool hermitian) =>
        Blas.Symv(uplo, n, alpha, a, lda, x, beta, y);

    protected override void Ger(int m, int n, double alpha, ReadOnlySpan<double> x, ReadOnlySpan<double> y, Span<double> a, int lda, bool conjugate) =>
        Blas.Ger(m, n, alpha, x, y, a, lda);

    protected override void Triangular(bool solve, BlasTriangle uplo, BlasTranspose trans, BlasDiagonal diag, int n, ReadOnlySpan<double> a, int lda, Span<double> x)
    {
        if (solve)
        {
            Blas.Trsv(uplo, trans, diag, n, a, lda, x);
        }
        else
        {
            Blas.Trmv(uplo, trans, diag, n, a, lda, x);
        }
    }
}

[Collection(nameof(ParallelismCollection))]
public sealed class SingleLevel2DeterminismTests : Level2DeterminismTests<float>
{
    protected override void ScaleMatrix(int m, int n, float beta, Span<float> a, int lda) =>
        Blas.Gemm(BlasTranspose.None, BlasTranspose.None, m, n, 0, 0f, [], m, [], 1, beta, a, lda);

    protected override void Gemv(BlasTranspose trans, int m, int n, float alpha, ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> x, float beta, Span<float> y) =>
        Blas.Gemv(trans, m, n, alpha, a, lda, x, beta, y);

    protected override void Symv(BlasTriangle uplo, int n, float alpha, ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> x, float beta, Span<float> y, bool hermitian) =>
        Blas.Symv(uplo, n, alpha, a, lda, x, beta, y);

    protected override void Ger(int m, int n, float alpha, ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<float> a, int lda, bool conjugate) =>
        Blas.Ger(m, n, alpha, x, y, a, lda);

    protected override void Triangular(bool solve, BlasTriangle uplo, BlasTranspose trans, BlasDiagonal diag, int n, ReadOnlySpan<float> a, int lda, Span<float> x)
    {
        if (solve)
        {
            Blas.Trsv(uplo, trans, diag, n, a, lda, x);
        }
        else
        {
            Blas.Trmv(uplo, trans, diag, n, a, lda, x);
        }
    }
}

[Collection(nameof(ParallelismCollection))]
public sealed class ComplexLevel2DeterminismTests : Level2DeterminismTests<Complex>
{
    private delegate void MatrixScale(Complex beta, int m, int n, Span<Complex> a, int lda);

    // 複素数の行列の係数倍には公開 BLAS の入口がないので、内部の総称カーネルを直接呼ぶ。
    private static readonly MatrixScale Scale = typeof(Matrix<Complex>).Assembly
        .GetType("Tuna.LinearAlgebra.ParallelLevel2`1")!.MakeGenericType(typeof(Complex))
        .GetMethod("ScaleOrClear", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic,
            null, [typeof(Complex), typeof(int), typeof(int), typeof(Span<Complex>), typeof(int)], null)!
        .CreateDelegate<MatrixScale>();

    protected override void ScaleMatrix(int m, int n, Complex beta, Span<Complex> a, int lda) =>
        Scale(beta, m, n, a, lda);

    protected override void Gemv(BlasTranspose trans, int m, int n, Complex alpha, ReadOnlySpan<Complex> a, int lda, ReadOnlySpan<Complex> x, Complex beta, Span<Complex> y) =>
        Blas.Gemv(trans, m, n, alpha, a, lda, x, beta, y);

    protected override void Symv(BlasTriangle uplo, int n, Complex alpha, ReadOnlySpan<Complex> a, int lda, ReadOnlySpan<Complex> x, Complex beta, Span<Complex> y, bool hermitian)
    {
        if (hermitian)
        {
            Blas.Hemv(uplo, n, alpha, a, lda, x, beta, y);
        }
        else
        {
            Blas.Symv(uplo, n, alpha, a, lda, x, beta, y);
        }
    }

    protected override void Ger(int m, int n, Complex alpha, ReadOnlySpan<Complex> x, ReadOnlySpan<Complex> y, Span<Complex> a, int lda, bool conjugate)
    {
        if (conjugate)
        {
            Blas.GerConjugate(m, n, alpha, x, y, a, lda);
        }
        else
        {
            Blas.Ger(m, n, alpha, x, y, a, lda);
        }
    }

    protected override void Triangular(bool solve, BlasTriangle uplo, BlasTranspose trans, BlasDiagonal diag, int n, ReadOnlySpan<Complex> a, int lda, Span<Complex> x)
    {
        if (solve)
        {
            Blas.Trsv(uplo, trans, diag, n, a, lda, x);
        }
        else
        {
            Blas.Trmv(uplo, trans, diag, n, a, lda, x);
        }
    }
}
