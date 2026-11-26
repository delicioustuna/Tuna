using System.Numerics;
using Tuna.LinearAlgebra;

namespace Tuna.Tests.LinearAlgebra;

[Collection(nameof(ParallelismCollection))]
public class BlasTests
{
    [Theory]
    [InlineData(32769, 8)] // Normal GEMV partitions rows, beyond the column-split threshold.
    [InlineData(257, 263)] // Normal GEMV partitions columns into private work buffers.
    public void ParallelGemvSupportsConcurrentCallsWithSeparateDestinations(int m, int n)
    {
        WithThreads(Environment.ProcessorCount, () =>
        {
            var a = new double[m * n];
            for (int j = 0; j < n; j++)
            {
                for (int i = 0; i < m; i++)
                {
                    a[i + j * m] = (i % 7 - j % 5) * 0.125;
                }
            }

            foreach (var trans in new[] { BlasTranspose.None, BlasTranspose.Transpose })
            {
                bool normal = trans == BlasTranspose.None;
                int rows = normal ? m : n, columns = normal ? n : m;
                var x = Enumerable.Range(0, columns).Select(i => (i % 9 - 4) * 0.25).ToArray();
                var expected = new double[rows];
                for (int i = 0; i < rows; i++)
                {
                    expected[i] = 0.75 * 2.0;
                    for (int j = 0; j < columns; j++)
                    {
                        expected[i] += -0.5 * a[normal ? i + j * m : j + i * m] * x[j];
                    }
                }

                var outputs = Enumerable.Range(0, 6).Select(_ => Enumerable.Repeat(2.0, rows).ToArray()).ToArray();
                Parallel.For(0, outputs.Length, index =>
                    Blas.Gemv(trans, m, n, -0.5, a, m, x, 0.75, outputs[index]));
                foreach (var output in outputs)
                {
                    Close(expected, output);
                }
            }
        });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SequentialGemvWithADestinationDoesNotAllocate(bool transpose)
    {
        WithThreads(1, () =>
        {
            var a = new Matrix<double>(9, 512);
            a.AsSpan().Fill(1.0);
            var x = new Tuna.LinearAlgebra.Vector<double>(transpose ? a.Rows : a.Columns);
            x.AsSpan().Fill(1.0);
            var destination = new Tuna.LinearAlgebra.Vector<double>(transpose ? a.Columns : a.Rows);
            ReadOnlyMatrixSpan<double> input = transpose ? a.T : a.AsMatrixSpan();
            for (int i = 0; i < 128; i++)
            {
                XMath.Multiply<double>(input, x, destination);
            }

            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 512; i++)
            {
                XMath.Multiply<double>(input, x, destination);
            }

            long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

            Assert.Equal(0L, allocated);
            Assert.All(destination.ToArray(), value => Assert.Equal((double)x.Length, value));
        });
    }

    public static TheoryData<int, int> VectorCases => new()
    {
        { 0, 1 }, { 1, 1 }, { 7, 1 }, { 31, 1 }, { 1025, 1 },
        { 300001, 1 }, { 300001, Environment.ProcessorCount },
    };

    public static TheoryData<int, int> MatrixCases => new()
    {
        { 1, 1 }, { 7, 1 }, { 65, 1 }, { 257, 1 },
        { 257, Environment.ProcessorCount }, { 1031, Environment.ProcessorCount },
    };

    [Theory, MemberData(nameof(VectorCases))]
    public void Level1AgainstScalarReference(int n, int threads)
    {
        WithThreads(threads, () =>
        {
            var random = new Random(321 + n);
            var x = Enumerable.Range(0, n).Select(_ => random.NextDouble() - .5).ToArray();
            var y = Enumerable.Range(0, n).Select(_ => random.NextDouble() - .5).ToArray();
            Near(x.Zip(y).Sum(p => p.First * p.Second), Blas.Dot(x, y));
            Near(Math.Sqrt(x.Sum(v => v * v)), Blas.Norm2(x));
            Near(x.Sum(Math.Abs), Blas.SumAbs(x));
            Assert.Equal(n == 0 ? -1 : Array.IndexOf(x, x.MaxBy(Math.Abs)), Blas.IndexOfMaxAbs(x));
            var actual = (double[])y.Clone();
            Blas.Axpy(1.25, x, actual);
            Close(x.Zip(y).Select(p => 1.25 * p.First + p.Second).ToArray(), actual);
            Blas.Scale(-2.0, actual);
            Close(x.Zip(y).Select(p => -2 * (1.25 * p.First + p.Second)).ToArray(), actual);
            Blas.Copy(x, actual);
            Assert.Equal(x, actual);
            var other = (double[])y.Clone();
            Blas.Swap(actual, other);
            Assert.Equal(y, actual);
            Assert.Equal(x, other);
            Blas.Rotate(actual, other, .8, .6);
            Close(x.Zip(y).Select(p => .8 * p.Second + .6 * p.First).ToArray(), actual);
            Close(x.Zip(y).Select(p => .8 * p.First - .6 * p.Second).ToArray(), other);

            var xf = x.Select(v => (float)v).ToArray();
            var yf = y.Select(v => (float)v).ToArray();
            Near(xf.Zip(yf).Sum(p => (double)p.First * p.Second), Blas.Dot(xf, yf), 2e-4);
            Near(Math.Sqrt(xf.Sum(v => (double)v * v)), Blas.Norm2(xf), 2e-6);
            Blas.Axpy(1.25f, xf, yf);
            for (int i = 0; i < n; i++)
            {
                Near(1.25 * xf[i] + (float)y[i], yf[i], 2e-6);
            }

            var xc = x.Zip(y).Select(p => new Complex(p.First, p.Second)).ToArray();
            var yc = y.Zip(x).Select(p => new Complex(p.First, -p.Second)).ToArray();
            var dot = Complex.Zero;
            var conjugateDot = Complex.Zero;
            for (int i = 0; i < n; i++)
            {
                dot += xc[i] * yc[i];
                conjugateDot += Complex.Conjugate(xc[i]) * yc[i];
            }

            Near(dot, Blas.Dot(xc, yc));
            Near(conjugateDot, Blas.DotConjugate(xc, yc));
            Near(Math.Sqrt(xc.Sum(v => v.Real * v.Real + v.Imaginary * v.Imaginary)), Blas.Norm2(xc));
            var alpha = new Complex(.7, -.3);
            var expected = xc.Zip(yc).Select(p => alpha * p.First + p.Second).ToArray();
            Blas.Axpy(alpha, xc, yc);
            for (int i = 0; i < n; i++)
            {
                Near(expected[i], yc[i]);
            }
        });
    }

    [Theory, MemberData(nameof(MatrixCases))]
    public void GemvAllTypesAndTransposesAgainstScalarReference(int n, int threads)
    {
        WithThreads(threads, () =>
        {
            int m = n + 3, lda = m + 5;
            var random = new Random(432 + n);
            var a = Enumerable.Range(0, lda * n).Select(_ => new Complex(random.NextDouble() - .5, random.NextDouble() - .5)).ToArray();
            var ad = a.Select(v => v.Real).ToArray();
            var af = ad.Select(v => (float)v).ToArray();
            var ai = ad.Select(v => (int)(v * 100)).ToArray();
            foreach (var trans in Enum.GetValues<BlasTranspose>())
            {
                bool normal = trans == BlasTranspose.None;
                int rows = normal ? m : n, cols = normal ? n : m;
                var x = Enumerable.Range(0, cols).Select(_ => new Complex(random.NextDouble() - .5, random.NextDouble() - .5)).ToArray();
                var xd = x.Select(v => v.Real).ToArray();
                var xf = xd.Select(v => (float)v).ToArray();
                var xi = xd.Select(v => (int)(v * 100)).ToArray();
                var yc = new Complex[rows];
                var yd = new double[rows];
                var yf = new float[rows];
                var yi = new int[rows];
                // beta=0 must ignore NaNs in destination.
                Array.Fill(yc, new Complex(double.NaN, double.NaN));
                Array.Fill(yd, double.NaN);
                Array.Fill(yf, float.NaN);
                Blas.Gemv(trans, m, n, Complex.One, a, lda, x, Complex.Zero, yc);
                Blas.Gemv(trans, m, n, 1.0, ad, lda, xd, 0.0, yd);
                Blas.Gemv(trans, m, n, 1f, af, lda, xf, 0f, yf);
                Blas.Gemv<int>(trans, m, n, 1, ai, lda, xi, 0, yi);
                for (int i = 0; i < rows; i++)
                {
                    Complex ec = 0;
                    double ed = 0, ef = 0;
                    int ei = 0;
                    for (int j = 0; j < cols; j++)
                    {
                        int index = normal ? i + j * lda : j + i * lda;
                        var v = trans == BlasTranspose.ConjugateTranspose ? Complex.Conjugate(a[index]) : a[index];
                        ec += v * x[j];
                        ed += ad[index] * xd[j];
                        ef += (double)af[index] * xf[j];
                        ei = unchecked(ei + ai[index] * xi[j]);
                    }

                    Near(ec, yc[i]);
                    Near(ed, yd[i]);
                    Near(ef, yf[i], 2e-4);
                    Assert.Equal(ei, yi[i]);
                }
            }
        });
    }

    [Theory, MemberData(nameof(MatrixCases))]
    public void TriangularMultiplyAndSolveAgainstKnownSolution(int n, int threads)
    {
        WithThreads(threads, () =>
        {
            int lda = n + 3;
            var random = new Random(543 + n);
            var a = Enumerable.Range(0, lda * n).Select(_ => (random.NextDouble() - .5) / n).ToArray();
            for (int i = 0; i < n; i++)
            {
                a[i + i * lda] = 2 + random.NextDouble();
            }

            var x = Enumerable.Range(0, n).Select(_ => random.NextDouble() - .5).ToArray();
            foreach (var uplo in Enum.GetValues<BlasTriangle>())
            {
                foreach (var trans in Enum.GetValues<BlasTranspose>())
                {
                    foreach (var diag in Enum.GetValues<BlasDiagonal>())
                    {
                        var expected = new double[n];
                        for (int i = 0; i < n; i++)
                        {
                            for (int j = 0; j < n; j++)
                            {
                                int row = trans == BlasTranspose.None ? i : j, col = trans == BlasTranspose.None ? j : i;
                                if (uplo == BlasTriangle.Lower ? row < col : row > col)
                                {
                                    continue;
                                }

                                expected[i] += (i == j && diag == BlasDiagonal.Unit ? 1 : a[row + col * lda]) * x[j];
                            }
                        }

                        var actual = (double[])x.Clone();
                        Blas.Trmv(uplo, trans, diag, n, a, lda, actual);
                        Close(expected, actual);
                        actual = (double[])expected.Clone();
                        Blas.Trsv(uplo, trans, diag, n, a, lda, actual);
                        Close(x, actual);
                    }
                }
            }
        });
    }

    [Theory, MemberData(nameof(MatrixCases))]
    public void SymmetricAndHermitianProductsAgainstScalarReference(int n, int threads)
    {
        WithThreads(threads, () =>
        {
            int lda = n + 3;
            var random = new Random(654 + n);
            var a = Enumerable.Range(0, lda * n).Select(_ => new Complex(random.NextDouble() - .5, random.NextDouble() - .5)).ToArray();
            var ad = a.Select(v => v.Real).ToArray();
            var x = Enumerable.Range(0, n).Select(_ => new Complex(random.NextDouble() - .5, random.NextDouble() - .5)).ToArray();
            var xd = x.Select(v => v.Real).ToArray();
            foreach (var uplo in Enum.GetValues<BlasTriangle>())
            {
                var yc = Enumerable.Repeat(new Complex(.3, -.2), n).ToArray();
                var yd = Enumerable.Repeat(.3, n).ToArray();
                var alpha = new Complex(.7, -.1);
                var beta = new Complex(-.4, .2);
                Blas.Hemv(uplo, n, alpha, a, lda, x, beta, yc);
                Blas.Symv(uplo, n, .7, ad, lda, xd, -.4, yd);
                for (int i = 0; i < n; i++)
                {
                    Complex ec = 0;
                    double ed = 0;
                    for (int j = 0; j < n; j++)
                    {
                        bool stored = uplo == BlasTriangle.Lower ? i >= j : i <= j;
                        int index = stored ? i + j * lda : j + i * lda;
                        var value = stored ? a[index] : Complex.Conjugate(a[index]);
                        if (i == j)
                        {
                            value = value.Real;
                        }

                        ec += value * x[j];
                        ed += ad[index] * xd[j];
                    }

                    Near(alpha * ec + beta * new Complex(.3, -.2), yc[i]);
                    Near(.7 * ed - .4 * .3, yd[i]);
                }
            }
        });
    }

    [Theory, MemberData(nameof(MatrixCases))]
    public void RankOneUpdateAgainstScalarReference(int n, int threads)
    {
        WithThreads(threads, () =>
        {
            // 縦長の行列にし、列の間隔の余り（行 m 以降）は書き換えないことも確かめる
            int m = n + 2, lda = m + 3;
            var rng = new Random(n);
            var ad = Enumerable.Range(0, lda * n).Select(_ => rng.NextDouble() - .5).ToArray();
            var xd = Enumerable.Range(0, m).Select(_ => rng.NextDouble() - .5).ToArray();
            var yd = Enumerable.Range(0, n).Select(_ => rng.NextDouble() - .5).ToArray();
            var ed = (double[])ad.Clone();
            for (int j = 0; j < n; j++)
            {
                for (int i = 0; i < m; i++)
                {
                    ed[i + j * lda] += -.6 * xd[i] * yd[j];
                }
            }

            Blas.Ger(m, n, -.6, xd, yd, ad, lda);
            Close(ed, ad);

            var alpha = new Complex(.4, -.9);
            var xc = xd.Select((v, i) => new Complex(v, yd[i % n])).ToArray();
            var yc = yd.Select((v, j) => new Complex(-v, xd[j])).ToArray();
            foreach (bool conjugate in new[] { false, true })
            {
                var ac = Enumerable.Range(0, lda * n).Select(_ => new Complex(rng.NextDouble() - .5, rng.NextDouble() - .5)).ToArray();
                var ec = (Complex[])ac.Clone();
                for (int j = 0; j < n; j++)
                {
                    for (int i = 0; i < m; i++)
                    {
                        ec[i + j * lda] += alpha * xc[i] * (conjugate ? Complex.Conjugate(yc[j]) : yc[j]);
                    }
                }

                if (conjugate)
                {
                    Blas.GerConjugate(m, n, alpha, xc, yc, ac, lda);
                }
                else
                {
                    Blas.Ger(m, n, alpha, xc, yc, ac, lda);
                }

                for (int k = 0; k < ac.Length; k++)
                {
                    Near(ec[k], ac[k]);
                }
            }
        });
    }

    public static TheoryData<string> Level2Operations => new()
    {
        "Ger", "GerConjugate", "Symv", "Hemv", "TrmvNone", "TrmvTranspose", "TrsvNone", "TrsvTranspose", "Syr", "Syr2",
    };

    [Theory, MemberData(nameof(Level2Operations))]
    public void SequentialLevel2OperationsDoNotAllocate(string operation)
    {
        WithThreads(1, () =>
        {
            // 三角行列は 64 列のブロックに分けて処理する大きさにする
            const int n = 200;
            var rng = new Random(7);
            var a = Enumerable.Range(0, n * n).Select(_ => (rng.NextDouble() - .5) * 1e-3).ToArray();
            for (int i = 0; i < n; i++)
            {
                a[i + i * n] = 1;
            }

            var ac = a.Select(v => new Complex(v, -v)).ToArray();
            var x = Enumerable.Range(0, n).Select(_ => rng.NextDouble() - .5).ToArray();
            var xc = x.Select(v => new Complex(v, v)).ToArray();
            var source = (double[])x.Clone();
            var y = new double[n];
            var yc = new Complex[n];
            Action call = operation switch
            {
                "Ger" => () => Blas.Ger(n, n, 1e-12, x, source, a, n),
                "GerConjugate" => () => Blas.GerConjugate(n, n, new Complex(1e-12, 0), xc, xc, ac, n),
                "Symv" => () => Blas.Symv(BlasTriangle.Lower, n, .7, a, n, x, 0.0, y),
                "Hemv" => () => Blas.Hemv(BlasTriangle.Upper, n, new Complex(.7, .1), ac, n, xc, Complex.Zero, yc),
                "TrmvNone" => () =>
                {
                    source.CopyTo(x, 0);
                    Blas.Trmv(BlasTriangle.Upper, BlasTranspose.None, BlasDiagonal.Unit, n, a, n, x);
                }
                ,
                "TrmvTranspose" => () =>
                {
                    source.CopyTo(x, 0);
                    Blas.Trmv(BlasTriangle.Lower, BlasTranspose.Transpose, BlasDiagonal.NonUnit, n, a, n, x);
                }
                ,
                "TrsvNone" => () =>
                {
                    source.CopyTo(x, 0);
                    Blas.Trsv(BlasTriangle.Lower, BlasTranspose.None, BlasDiagonal.NonUnit, n, a, n, x);
                }
                ,
                "TrsvTranspose" => () =>
                {
                    source.CopyTo(x, 0);
                    Blas.Trsv(BlasTriangle.Upper, BlasTranspose.Transpose, BlasDiagonal.Unit, n, a, n, x);
                }
                ,
                "Syr" => () => Blas.Syr(BlasTriangle.Lower, n, 1e-12, x, a, n),
                "Syr2" => () => Blas.Syr2(BlasTriangle.Upper, n, 1e-12, x, source, a, n),
                _ => throw new ArgumentOutOfRangeException(nameof(operation)),
            };
            for (int i = 0; i < 64; i++)
            {
                call();
            }

            // 段階的 JIT の切り替えの途中で、実行時が一度だけ数 KB を割り当てることがある（変更していない Syr でも起きる）。
            // 呼び出しごとの割り当てならどの回も 0 にならないので、何回か測って 0 の回があればよい
            long allocated = long.MaxValue;
            for (int round = 0; round < 5 && allocated != 0; round++)
            {
                long before = GC.GetAllocatedBytesForCurrentThread();
                for (int i = 0; i < 256; i++)
                {
                    call();
                }

                allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            }

            Assert.Equal(0L, allocated);
        });
    }

    public static TheoryData<string> Level1Operations => new()
    {
        "Dot", "Axpy", "Scale", "SumAbs", "Rotate", "Swap", "Copy", "Norm2",
    };

    [Theory, MemberData(nameof(Level1Operations))]
    public void SequentialLevel1OperationsDoNotAllocate(string operation)
    {
        WithThreads(1, () =>
        {
            // 並列化する大きさ（1スレッドあたり 32K 要素）を超える長さにする
            const int n = 1 << 16;
            var x = Enumerable.Range(0, n).Select(i => (i % 11 - 5) * 1e-3).ToArray();
            var y = Enumerable.Range(0, n).Select(i => (i % 7 - 3) * 1e-3).ToArray();
            double sink = 0;
            Action call = operation switch
            {
                "Dot" => () => sink += Blas.Dot(x, y),
                "Axpy" => () => Blas.Axpy(1e-12, x, y),
                "Scale" => () => Blas.Scale(1.0, x),
                "SumAbs" => () => sink += Blas.SumAbs(x),
                "Rotate" => () => Blas.Rotate(x, y, 1.0, 0.0),
                "Swap" => () => Blas.Swap(x, y),
                "Copy" => () => Blas.Copy(x, y),
                "Norm2" => () => sink += Blas.Norm2(x),
                _ => throw new ArgumentOutOfRangeException(nameof(operation)),
            };
            for (int i = 0; i < 64; i++)
            {
                call();
            }

            // 段階的 JIT の切り替えの途中の一度だけの割り当ては除く（SequentialLevel2OperationsDoNotAllocate と同じ）
            long allocated = long.MaxValue;
            for (int round = 0; round < 5 && allocated != 0; round++)
            {
                long before = GC.GetAllocatedBytesForCurrentThread();
                for (int i = 0; i < 256; i++)
                {
                    call();
                }

                allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            }

            Assert.Equal(0L, allocated);
        });
    }

    [Fact]
    public void NormExtremeValuesAndArgumentChecks()
    {
        Near(Math.Sqrt(2) * 1e300, Blas.Norm2(new double[] { 1e300, -1e300 }));
        Assert.InRange(Blas.Norm2(new double[] { 1e-300, -1e-300 }) / 1e-300, 1.41421356237309, 1.41421356237310);
        Assert.Throws<ArgumentException>(() => Blas.Dot(new double[2], new double[3]));
        Assert.Throws<ArgumentOutOfRangeException>(() => Blas.Gemv(BlasTranspose.None, 3, 3, 1.0, new double[9], 2, new double[3], 0.0, new double[3]));
    }

    private static void WithThreads(int threads, Action action)
    {
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = threads;
            action();
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    private static void Close(double[] expected, double[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Near(expected[i], actual[i]);
        }
    }

    private static void Near(double expected, double actual, double tolerance = 2e-10) =>
        Assert.True(double.IsFinite(actual) && Math.Abs(expected - actual) <= tolerance * Math.Max(1, Math.Abs(expected)), $"expected={expected:R}, actual={actual:R}");

    private static void Near(Complex expected, Complex actual) =>
        Assert.True(double.IsFinite(actual.Magnitude) && (expected - actual).Magnitude <= 2e-10 * Math.Max(1, expected.Magnitude), $"expected={expected}, actual={actual}");
}
