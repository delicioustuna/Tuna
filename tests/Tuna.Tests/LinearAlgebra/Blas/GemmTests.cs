using Tuna.LinearAlgebra;
using static Tuna.Tests.LinearAlgebra.LinearAlgebraTestUtilities;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>
/// 行列積 <c>C ← α·op(A)·op(B) + β·C</c>（<c>Blas.Gemm</c> の単精度・倍精度）の動作確認。
/// 参照は倍精度の単純な三重ループで求め、誤差は <c>|α|·Σ|a·b| + |β·c|</c> に対する丸め誤差の上界で判定する
/// </summary>
[Collection(nameof(ParallelismCollection))]
public class GemmTests
{
    /// <summary>単精度の丸めの単位の2倍（<c>2⁻²³</c>）</summary>
    private const double SingleEpsilon = 1.1920928955078125e-7;

    /// <summary>列の間隔の余りに入れておく値（読み書きしないことを確かめる）</summary>
    private const double Gap = 12345;

    public static TheoryData<int, int, int> Sizes => new()
    {
        { 1, 1, 1 },
        { 3, 70, 40 },      // m < 4 はパッキングしない
        { 7, 5, 9 },        // m·k が小さい積もパッキングしない
        { 24, 9, 24 },      // m·k = 576（パッキングしない上限）
        { 25, 9, 24 },      // パッキングする最小に近い大きさ
        { 33, 31, 257 },    // 端のパネルと、KC（256）をまたぐ k
        { 49, 17, 70 },
        { 97, 70, 300 },
    };

    /// <summary>(α, β) の組。β = 0 では C に、α = 0 では A・B に NaN を入れて、読まないことを確かめる</summary>
    private static readonly (double Alpha, double Beta)[] Coefficients =
    [
        (1.25, 0),
        (-0.5, 1),
        (0.75, -1.5),
        (0, 0.5),
        (0, 0),
    ];

    [Theory]
    [MemberData(nameof(Sizes))]
    public void DoubleMatchesReference(int m, int n, int k)
    {
        foreach (var transA in Enum.GetValues<BlasTranspose>())
        {
            foreach (var transB in Enum.GetValues<BlasTranspose>())
            {
                foreach (var (alpha, beta) in Coefficients)
                {
                    var p = Operands.Create(transA, transB, m, n, k, alpha, beta, 1000 * m + 10 * n + k);
                    var c = RunDouble(p);
                    p.Verify(c, Epsilon, $"double {transA}·{transB} α={alpha} β={beta}");
                }
            }
        }
    }

    [Theory]
    [MemberData(nameof(Sizes))]
    public void SingleMatchesReference(int m, int n, int k)
    {
        foreach (var transA in Enum.GetValues<BlasTranspose>())
        {
            foreach (var transB in Enum.GetValues<BlasTranspose>())
            {
                foreach (var (alpha, beta) in Coefficients)
                {
                    var p = Operands.Create(transA, transB, m, n, k, alpha, beta, 1000 * m + 10 * n + k);
                    var c = RunSingle(p);
                    p.Verify(c, SingleEpsilon, $"float {transA}·{transB} α={alpha} β={beta}");
                }
            }
        }
    }

    /// <summary>結果はスレッド数によらずビット単位で同じ。複数のスレッドから同時に呼んでも同じ（並列区間が使えない呼び出しは逐次に計算する）</summary>
    [Theory]
    [InlineData(260, 200, 300)]   // 積を複数のスレッドに分け、倍精度では C の β 倍も列で分ける
    [InlineData(400, 200, 64)]    // 単精度でも C の β 倍を列で分ける
    public void ResultsDoNotDependOnThreadCount(int m, int n, int k)
    {
        var p = Operands.Create(BlasTranspose.Transpose, BlasTranspose.None, m, n, k, 0.75, -1.5, m + n + k);
        var sequential = WithThreads(1, () => RunDouble(p));
        var sequentialSingle = WithThreads(1, () => RunSingle(p));
        p.Verify(sequential, Epsilon, "double");
        p.Verify(sequentialSingle, SingleEpsilon, "float");
        WithThreads(Environment.ProcessorCount, () =>
        {
            Assert.Equal(Bits(sequential), Bits(RunDouble(p)));
            Assert.Equal(Bits(sequentialSingle), Bits(RunSingle(p)));
            var outputs = new double[4][];
            var singles = new double[4][];
            Parallel.For(0, outputs.Length, i =>
            {
                outputs[i] = RunDouble(p);
                singles[i] = RunSingle(p);
            });
            Assert.All(outputs, output => Assert.Equal(Bits(sequential), Bits(output)));
            Assert.All(singles, output => Assert.Equal(Bits(sequentialSingle), Bits(output)));
        });

        static long[] Bits(double[] x) => Array.ConvertAll(x, BitConverter.DoubleToInt64Bits);
    }

    /// <summary>
    /// β = 0（-0 を含む）なら C の元の値（NaN・∞）は読まない。β が 0 でなければ読むので、NaN・∞ は残る
    /// </summary>
    [Fact]
    public void ZeroBetaIgnoresNonFiniteValuesOfC()
    {
        // A = [[1, 3], [2, 4]]、B は単位行列なので α·A·B = 2A
        var a = new double[] { 1, 2, 3, 4 };
        var b = new double[] { 1, 0, 0, 1 };
        foreach (double beta in new[] { 0.0, -0.0 })
        {
            var c = new double[] { double.NaN, double.PositiveInfinity, Gap, double.NegativeInfinity, double.NaN };
            Blas.Gemm(BlasTranspose.None, BlasTranspose.None, 2, 2, 2, 2.0, a, 2, b, 2, beta, c, 3);
            Assert.Equal(new double[] { 2, 4, Gap, 6, 8 }, c);

            var cf = new float[] { float.NaN, float.PositiveInfinity, (float)Gap, float.NegativeInfinity, float.NaN };
            Blas.Gemm(BlasTranspose.None, BlasTranspose.None, 2, 2, 2, 2f, Array.ConvertAll(a, v => (float)v), 2, Array.ConvertAll(b, v => (float)v), 2, (float)beta, cf, 3);
            Assert.Equal(new float[] { 2, 4, (float)Gap, 6, 8 }, cf);
        }

        var kept = new double[] { double.NaN, double.PositiveInfinity };
        Blas.Gemm(BlasTranspose.None, BlasTranspose.None, 2, 1, 2, 1.0, a, 2, b, 2, 0.5, kept, 2);
        Assert.True(double.IsNaN(kept[0]));
        Assert.Equal(double.PositiveInfinity, kept[1]);
    }

    /// <summary>k = 0 なら C ← β·C（A・B は空でよい）。m か n が 0 なら何もしない</summary>
    [Fact]
    public void EmptyProducts()
    {
        var c = new double[] { 1, 2, Gap, 3, 4 };
        Blas.Gemm(BlasTranspose.None, BlasTranspose.None, 2, 2, 0, 1.0, ReadOnlySpan<double>.Empty, 2, ReadOnlySpan<double>.Empty, 1, -2.0, c, 3);
        Assert.Equal(new double[] { -2, -4, Gap, -6, -8 }, c);
        Blas.Gemm(BlasTranspose.Transpose, BlasTranspose.Transpose, 2, 2, 0, 1.0, ReadOnlySpan<double>.Empty, 1, ReadOnlySpan<double>.Empty, 2, 1.0, c, 3);
        Assert.Equal(new double[] { -2, -4, Gap, -6, -8 }, c);

        var cf = new float[] { float.NaN, float.NaN, (float)Gap, float.NaN, float.NaN };
        Blas.Gemm(BlasTranspose.None, BlasTranspose.None, 2, 2, 0, 1f, ReadOnlySpan<float>.Empty, 2, ReadOnlySpan<float>.Empty, 1, 0f, cf, 3);
        Assert.Equal(new float[] { 0, 0, (float)Gap, 0, 0 }, cf);

        Blas.Gemm(BlasTranspose.None, BlasTranspose.None, 0, 3, 4, 1.0, ReadOnlySpan<double>.Empty, 1, new double[12], 4, 0.0, Span<double>.Empty, 1);
        Blas.Gemm(BlasTranspose.None, BlasTranspose.None, 3, 0, 4, 1f, new float[12], 3, ReadOnlySpan<float>.Empty, 4, 0f, Span<float>.Empty, 3);
    }

    /// <summary>出力が入力と要素を1つでも共有すれば例外にし、C は書き換えない（A と B は重なってよい）</summary>
    [Fact]
    public void OutputOverlappingAnInputIsRejected()
    {
        var data = Enumerable.Range(0, 100).Select(i => (double)i).ToArray();
        var original = (double[])data.Clone();
        var other = new double[100];

        // その場での積（C が A そのもの、C が B そのもの）
        AssertOverlap(() => Blas.Gemm(BlasTranspose.None, BlasTranspose.None, 10, 10, 10, 1.0, data, 10, other, 10, 0.0, data, 10));
        AssertOverlap(() => Blas.Gemm(BlasTranspose.None, BlasTranspose.Transpose, 10, 10, 10, 1.0, other, 10, data, 10, 0.0, data, 10));

        // C の最後の要素が B の最初の要素（共有するのはこの1要素だけ）、C の最初の要素が転置した A の最後の要素
        AssertOverlap(() => Blas.Gemm(BlasTranspose.None, BlasTranspose.None, 9, 9, 1, 1.0, other, 9, data.AsSpan(88), 1, 0.0, data, 10));
        AssertOverlap(() => Blas.Gemm(BlasTranspose.Transpose, BlasTranspose.None, 2, 2, 3, 1.0, data, 3, other, 3, 0.0, data.AsSpan(5), 10));

        // 単精度も同じ
        var single = Array.ConvertAll(data, v => (float)v);
        AssertOverlap(() => Blas.Gemm(BlasTranspose.None, BlasTranspose.None, 10, 10, 10, 1f, single, 10, single, 10, 0f, single, 10));
        Assert.Equal(Array.ConvertAll(original, v => (float)v), single);
        Assert.Equal(original, data);

        // A と B は同じ領域でよい（C ← A·Aᵀ）
        var p = Operands.Create(BlasTranspose.None, BlasTranspose.Transpose, 5, 5, 3, 1, 0, 7);
        var gram = new double[25];
        Blas.Gemm(BlasTranspose.None, BlasTranspose.Transpose, 5, 5, 3, 1.0, p.A, p.Lda, p.A, p.Lda, 0.0, gram, 5);
        for (int j = 0; j < 5; j++)
        {
            for (int i = 0; i < 5; i++)
            {
                double sum = 0;
                for (int q = 0; q < 3; q++)
                {
                    sum += p.A[i + q * p.Lda] * p.A[j + q * p.Lda];
                }

                Assert.Equal(sum, gram[i + j * 5], 1e-15);
            }
        }

        static void AssertOverlap(Action call) => Assert.Equal("c", Assert.Throws<ArgumentException>(call).ParamName);
    }

    /// <summary>同じ配列の部分行列は、範囲が重なっても要素を共有しなければ計算する</summary>
    [Fact]
    public void DisjointSubmatricesOfOneArrayAreAccepted()
    {
        // LU 分解の更新と同じ形 A₂₂ ← A₂₂ − A₂₁·A₁₂（列優先の 12 × 12 を 4 で分ける）。A₁₂ と A₂₂ は範囲が重なるが要素を共有しない
        const int n = 12, s = 4;
        var random = new Random(3);
        var data = Enumerable.Range(0, n * n).Select(_ => 2 * random.NextDouble() - 1).ToArray();
        var expected = (double[])data.Clone();
        for (int j = s; j < n; j++)
        {
            for (int i = s; i < n; i++)
            {
                for (int q = 0; q < s; q++)
                {
                    expected[i + j * n] -= data[i + q * n] * data[q + j * n];
                }
            }
        }

        Blas.Gemm(BlasTranspose.None, BlasTranspose.None, n - s, n - s, s, -1.0, data.AsSpan(s), n, data.AsSpan(s * n), n, 1.0, data.AsSpan(s + s * n), n);
        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(expected[i], data[i], 1e-14);
        }

        // 列の間隔 2m の配列の上半分の行を A、下半分の行を C にする（A と C の要素は列ごとに交互に並ぶ）
        const int m = 30, k = 40, columns = 20;
        var shared = new float[2 * m * Math.Max(k, columns)];
        for (int q = 0; q < k; q++)
        {
            for (int i = 0; i < m; i++)
            {
                shared[i + q * 2 * m] = (float)(2 * random.NextDouble() - 1);
            }
        }

        var b = Enumerable.Range(0, k * columns).Select(_ => (float)(2 * random.NextDouble() - 1)).ToArray();
        Blas.Gemm(BlasTranspose.None, BlasTranspose.None, m, columns, k, 1f, shared, 2 * m, b, k, 0f, shared.AsSpan(m), 2 * m);
        for (int j = 0; j < columns; j++)
        {
            for (int i = 0; i < m; i++)
            {
                double sum = 0, bound = 0;
                for (int q = 0; q < k; q++)
                {
                    double t = (double)shared[i + q * 2 * m] * b[q + j * k];
                    sum += t;
                    bound += Math.Abs(t);
                }

                Assert.True(Math.Abs(shared[m + i + j * 2 * m] - sum) <= (k + 4) * SingleEpsilon * bound, $"[{i}, {j}]");
            }
        }
    }

    /// <summary>大きさ・列の間隔・領域の長さの検査（引数名で報告する）</summary>
    [Fact]
    public void ArgumentChecks()
    {
        var a = new double[12];
        var b = new double[12];
        var c = new double[12];
        Assert.Equal("m", OutOfRange(() => Blas.Gemm(BlasTranspose.None, BlasTranspose.None, -1, 3, 4, 1.0, a, 3, b, 4, 0.0, c, 3)));
        Assert.Equal("n", OutOfRange(() => Blas.Gemm(BlasTranspose.None, BlasTranspose.None, 3, -1, 4, 1.0, a, 3, b, 4, 0.0, c, 3)));
        Assert.Equal("k", OutOfRange(() => Blas.Gemm(BlasTranspose.None, BlasTranspose.None, 3, 3, -1, 1.0, a, 3, b, 4, 0.0, c, 3)));

        // 列の間隔は格納した行列の行数以上（転置した A は k 行、転置した B は n 行）、行数が 0 でも 1 以上
        Assert.Equal("lda", OutOfRange(() => Blas.Gemm(BlasTranspose.None, BlasTranspose.None, 3, 3, 4, 1.0, a, 2, b, 4, 0.0, c, 3)));
        Assert.Equal("lda", OutOfRange(() => Blas.Gemm(BlasTranspose.Transpose, BlasTranspose.None, 3, 3, 4, 1.0, a, 3, b, 4, 0.0, c, 3)));
        Assert.Equal("ldb", OutOfRange(() => Blas.Gemm(BlasTranspose.None, BlasTranspose.None, 3, 3, 4, 1.0, a, 3, b, 3, 0.0, c, 3)));
        Assert.Equal("ldb", OutOfRange(() => Blas.Gemm(BlasTranspose.None, BlasTranspose.ConjugateTranspose, 3, 4, 3, 1.0, a, 3, b, 3, 0.0, c, 3)));
        Assert.Equal("ldc", OutOfRange(() => Blas.Gemm(BlasTranspose.None, BlasTranspose.None, 3, 3, 4, 1.0, a, 3, b, 4, 0.0, c, 2)));
        Assert.Equal("ldc", OutOfRange(() => Blas.Gemm(BlasTranspose.None, BlasTranspose.None, 0, 3, 4, 1.0, a, 1, b, 4, 0.0, c, 0)));

        // 領域が短い
        Assert.Equal("a", Short(() => Blas.Gemm(BlasTranspose.None, BlasTranspose.None, 3, 3, 4, 1.0, a.AsSpan(1), 3, b, 4, 0.0, c, 3)));
        Assert.Equal("b", Short(() => Blas.Gemm(BlasTranspose.None, BlasTranspose.None, 3, 3, 4, 1.0, a, 3, b.AsSpan(1), 4, 0.0, c, 3)));
        Assert.Equal("c", Short(() => Blas.Gemm(BlasTranspose.None, BlasTranspose.None, 3, 5, 4, 1.0, a, 3, new double[20], 4, 0.0, c, 3)));

        // 単精度も同じ検査を通る（転置した A は 4 × 3 で、列の間隔 5 なら 14 要素が要る）
        var af = new float[12];
        Assert.Equal("ldc", OutOfRange(() => Blas.Gemm(BlasTranspose.None, BlasTranspose.None, 3, 3, 4, 1f, af, 3, new float[12], 4, 0f, new float[12], 2)));
        Assert.Equal("a", Short(() => Blas.Gemm(BlasTranspose.Transpose, BlasTranspose.None, 3, 3, 4, 1f, af, 5, new float[12], 4, 0f, new float[9], 3)));

        static string? OutOfRange(Action call) => Assert.Throws<ArgumentOutOfRangeException>(call).ParamName;

        static string? Short(Action call) => Assert.Throws<ArgumentException>(call).ParamName;
    }

    /// <summary>逐次の計算では、パッキングする積もしない積も呼び出しごとに割り当てない</summary>
    [Theory]
    [InlineData(5, 7, 9)]
    [InlineData(64, 64, 64)]
    public void SequentialGemmDoesNotAllocate(int m, int n, int k)
    {
        WithThreads(1, () =>
        {
            var p = Operands.Create(BlasTranspose.None, BlasTranspose.Transpose, m, n, k, 0.5, 0.25, 9);
            var a = Array.ConvertAll(p.A, v => (float)v);
            var b = Array.ConvertAll(p.B, v => (float)v);
            var c = (double[])p.C.Clone();
            var cf = Array.ConvertAll(p.C, v => (float)v);
            Action call = () =>
            {
                Blas.Gemm(BlasTranspose.None, BlasTranspose.Transpose, m, n, k, 1e-3, p.A, p.Lda, p.B, p.Ldb, 0.25, c, p.Ldc);
                Blas.Gemm(BlasTranspose.None, BlasTranspose.Transpose, m, n, k, 1e-3f, a, p.Lda, b, p.Ldb, 0.25f, cf, p.Ldc);
            };
            for (int i = 0; i < 64; i++)
            {
                call();
            }

            // 段階的 JIT の切り替えの途中の一度だけの割り当ては除く（BlasTests.SequentialLevel2OperationsDoNotAllocate と同じ）
            long allocated = long.MaxValue;
            for (int round = 0; round < 5 && allocated != 0; round++)
            {
                long before = GC.GetAllocatedBytesForCurrentThread();
                for (int i = 0; i < 64; i++)
                {
                    call();
                }

                allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            }

            Assert.Equal(0L, allocated);
        });
    }

    private static double[] RunDouble(Operands p)
    {
        var c = (double[])p.C.Clone();
        Blas.Gemm(p.TransA, p.TransB, p.M, p.N, p.K, p.Alpha, p.A, p.Lda, p.B, p.Ldb, p.Beta, c, p.Ldc);
        return c;
    }

    /// <summary>同じ値の単精度の積（結果は比べやすいように倍精度に戻す）</summary>
    private static double[] RunSingle(Operands p)
    {
        var c = Array.ConvertAll(p.C, v => (float)v);
        Blas.Gemm(p.TransA, p.TransB, p.M, p.N, p.K, (float)p.Alpha, Array.ConvertAll(p.A, v => (float)v), p.Lda,
            Array.ConvertAll(p.B, v => (float)v), p.Ldb, (float)p.Beta, c, p.Ldc);
        return Array.ConvertAll(c, v => (double)v);
    }

    private static void WithThreads(int threads, Action action) => WithThreads(threads, () =>
    {
        action();
        return 0;
    });

    private static T WithThreads<T>(int threads, Func<T> action)
    {
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = threads;
            return action();
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    /// <summary>
    /// 列の間隔に余りのある A・B・C（領域は最小の長さ）。値は単精度で表せるので、単精度の積の参照にもそのまま使う。
    /// 余りには <see cref="Gap"/> を入れ、α = 0 なら A・B を、β = 0 なら C を NaN にする（読まなければ結果に出ない）
    /// </summary>
    private sealed record Operands(
        BlasTranspose TransA, BlasTranspose TransB, int M, int N, int K, double Alpha, double Beta,
        double[] A, int Lda, double[] B, int Ldb, double[] C, int Ldc)
    {
        internal static Operands Create(BlasTranspose transA, BlasTranspose transB, int m, int n, int k, double alpha, double beta, int seed)
        {
            bool ta = transA != BlasTranspose.None, tb = transB != BlasTranspose.None;
            var random = new Random(seed);
            double? unread = double.NaN;
            var (a, lda) = Fill(ta ? k : m, ta ? m : k, 2, alpha == 0 ? unread : null, random);
            var (b, ldb) = Fill(tb ? n : k, tb ? k : n, 3, alpha == 0 ? unread : null, random);
            var (c, ldc) = Fill(m, n, 1, beta == 0 ? unread : null, random);
            return new(transA, transB, m, n, k, alpha, beta, a, lda, b, ldb, c, ldc);
        }

        /// <summary>
        /// <c>rows × columns</c> の行列を、列の間隔 <c>rows + extra</c> で最小の長さの配列に入れる。
        /// 要素は <paramref name="value"/>（null なら単精度で表せる [-1, 1) の乱数）
        /// </summary>
        private static (double[] Data, int Ld) Fill(int rows, int columns, int extra, double? value, Random random)
        {
            int ld = rows + extra;
            var data = new double[(columns - 1) * ld + rows];
            for (int t = 0; t < data.Length; t++)
            {
                data[t] = t % ld < rows ? value ?? (float)(2 * random.NextDouble() - 1) : Gap;
            }

            return (data, ld);
        }

        /// <summary>各要素が参照と <c>(k + 4)·epsilon·(|α|·Σ|a·b| + |β·c|)</c> 以内で一致し、列の間隔の余りは元のまま</summary>
        internal void Verify(double[] actual, double epsilon, string label)
        {
            bool ta = TransA != BlasTranspose.None, tb = TransB != BlasTranspose.None;
            Assert.Equal(C.Length, actual.Length);
            for (int index = 0; index < actual.Length; index++)
            {
                int i = index % Ldc, j = index / Ldc;
                if (i >= M)
                {
                    Assert.Equal(Gap, actual[index]);
                    continue;
                }

                double sum = 0, bound = 0;
                if (Alpha != 0)
                {
                    for (int q = 0; q < K; q++)
                    {
                        double t = (ta ? A[q + i * Lda] : A[i + q * Lda]) * (tb ? B[j + q * Ldb] : B[q + j * Ldb]);
                        sum += t;
                        bound += Math.Abs(t);
                    }
                }

                double old = Beta == 0 ? 0 : C[index];
                double expected = Alpha * sum + Beta * old;
                double tolerance = (K + 4) * epsilon * (Math.Abs(Alpha) * bound + Math.Abs(Beta * old));
                Assert.True(Math.Abs(actual[index] - expected) <= tolerance, $"{label} [{i}, {j}]: {actual[index]:R} vs {expected:R}");
            }
        }
    }
}
