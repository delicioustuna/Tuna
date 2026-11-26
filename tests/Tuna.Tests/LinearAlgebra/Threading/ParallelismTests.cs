using Tuna.LinearAlgebra;
using static Tuna.Tests.LinearAlgebra.LinearAlgebraTestUtilities;

namespace Tuna.Tests.LinearAlgebra;

// 並列化した経路（行列積の分割、行列ベクトル積、分割統治法の部分問題と永年方程式、固有ベクトルの後退代入、
// バルジ追跡の先読み）が単一スレッドと同じ精度の結果を返すことを確認する。
// 並列化は n ≳ 100 で有効になるので、閾値・ブロック幅をまたぐ大きさで比べる。
[Trait("LinearAlgebra", "Parallelism")]
[Collection(nameof(ParallelismCollection))]
public class ParallelismTests
{
    public static TheoryData<int> ParallelSizes => new() { 130, 300, 520 };

    [Fact]
    public void RejectsNonPositiveDegree()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => LinearAlgebraSettings.MaxDegreeOfParallelism = 0);
    }

    /// <summary>休止と再開、および参加人数の切り替えで作業を取りこぼさない。</summary>
    [Fact]
    public async Task WorkersResumeAfterParkingAndChangingTeamSize()
    {
        if (Environment.ProcessorCount < 2)
        {
            return;
        }

        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            int maxThreads = Environment.ProcessorCount;
            LinearAlgebraSettings.MaxDegreeOfParallelism = maxThreads;
            // 演算カーネルの閾値に依存せず、実際の並列実行器を負荷・休止させる。
            var type = typeof(Matrix<double>).Assembly.GetType("Tuna.LinearAlgebra.Parallelism")!;
            // 同じ名前の、値型の作業を受け取る版と区別する
            var method = type.GetMethods(System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)
                .Single(m => m.Name == "ForStatic" && !m.IsGenericMethod);
            var run = method.CreateDelegate<Action<int, Action<int>>>();
            var work = Task.Run(() =>
            {
                var visits = new int[maxThreads];
                for (int iteration = 0; iteration < 5000; iteration++)
                {
                    if (iteration % 8 == 0)
                    {
                        Thread.Sleep(3); // 作業スレッドの2msスピン期間を超える
                    }

                    int degree = iteration % 3 == 0 ? 2 : maxThreads;
                    Array.Clear(visits);
                    run(degree, index => Interlocked.Increment(ref visits[index]));
                    for (int i = 0; i < visits.Length; i++)
                    {
                        Assert.Equal(i < degree ? 1 : 0, visits[i]);
                    }
                }
            });
            // 起床を取りこぼした場合、テストプロセス全体を無期限に止めない。
            await work.WaitAsync(TimeSpan.FromSeconds(60));
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    /// <summary>単一スレッドと並列で、どちらも残差が小さく、分解の値（対角・固有値・特異値）が丸め誤差の範囲で一致する</summary>
    [Theory]
    [MemberData(nameof(ParallelSizes))]
    public void CholeskyMatchesSingleThread(int n)
    {
        var a = RandomPositiveDefinite(n, 700 + n);
        var (one, many) = RunBoth(() => a.Cholesky());
        foreach (var c in new[] { one, many })
        {
            AssertLowerTriangular(c.L);
            AssertRatio(Difference(a, MultiplyNaive(c.L, c.L, transposeB: true)), Norm(a), n, "A - LLᵀ");
        }

        AssertClose(Diagonal(one.L), Diagonal(many.L), Norm(a), n);
    }

    [Theory]
    [MemberData(nameof(ParallelSizes))]
    public void QrMatchesSingleThread(int n)
    {
        var a = Random(n, n, 710 + n);
        var (one, many) = RunBoth(() => a.Qr());
        foreach (var qr in new[] { one, many })
        {
            AssertRatio(Difference(a, MultiplyNaive(qr.Q, qr.R)), Norm(a), n, "A - QR");
            AssertRatio(OrthogonalityError(qr.Q), 1, n, "QᵀQ - I");
        }

        AssertClose(Diagonal(one.R), Diagonal(many.R), Norm(a), n);
    }

    [Theory]
    [MemberData(nameof(ParallelSizes))]
    public void SymmetricEigenMatchesSingleThread(int n)
    {
        var a = RandomSymmetric(n, 720 + n);
        var (one, many) = RunBoth(() => a.SymmetricEigen());
        foreach (var evd in new[] { one, many })
        {
            var v = evd.Vectors!;
            AssertRatio(Difference(MultiplyNaive(a, v), MultiplyNaive(v, Matrix<double>.FromDiagonal(evd.Values))), Norm(a), n, "AV - VΛ");
            AssertRatio(OrthogonalityError(v), 1, n, "VᵀV - I");
        }

        AssertClose(one.Values, many.Values, Norm(a), n);
    }

    [Theory]
    [MemberData(nameof(ParallelSizes))]
    public void EigenMatchesSingleThread(int n)
    {
        var a = Random(n, n, 730 + n);
        var (one, many) = RunBoth(() => a.Eigen());
        foreach (var evd in new[] { one, many })
        {
            AssertRatio(EigenResidual(a, evd), Norm(a), n, "AV - VD");
        }

        // 収束の順序が変わりうるので、固有値は並べ替えてから比べる（乱数行列の固有値は良条件）
        var x = one.Values.OrderBy(c => c.Real).ThenBy(c => c.Imaginary).ToArray();
        var y = many.Values.OrderBy(c => c.Real).ThenBy(c => c.Imaginary).ToArray();
        for (int i = 0; i < n; i++)
        {
            Assert.True((x[i] - y[i]).Magnitude <= 1e-9 * Norm(a), $"λ[{i}]: {x[i]} vs {y[i]}");
        }
    }

    [Theory]
    [InlineData(130, 130)]
    [InlineData(300, 300)]
    [InlineData(520, 520)]
    [InlineData(700, 300)]
    public void SvdMatchesSingleThread(int m, int n)
    {
        var a = Random(m, n, 740 + m + n);
        var (one, many) = RunBoth(() => a.Svd());
        foreach (var svd in new[] { one, many })
        {
            var usv = MultiplyNaive(MultiplyNaive(svd.U!, Matrix<double>.FromDiagonal(svd.Values)), svd.V!, transposeB: true);
            AssertRatio(Difference(a, usv), Norm(a), Math.Max(m, n), "A - UΣVᵀ");
            AssertRatio(OrthogonalityError(svd.U!), 1, m, "UᵀU - I");
            AssertRatio(OrthogonalityError(svd.V!), 1, n, "VᵀV - I");
        }

        AssertClose(one.Values, many.Values, Norm(a), Math.Max(m, n));
    }

    /// <summary>複数のスレッドから同時に分解しても正しい（作業スレッドを確保できなかった呼び出しは逐次に計算される）</summary>
    [Fact]
    public void ConcurrentCallsAreCorrect()
    {
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = Environment.ProcessorCount;
            const int n = 260;
            var inputs = Enumerable.Range(0, 8).Select(i => RandomSymmetric(n, 750 + i)).ToArray();
            var results = new SymmetricEigenDecomposition[inputs.Length];
            var svds = new SingularValueDecomposition[inputs.Length];
            Parallel.For(0, inputs.Length, new ParallelOptions { MaxDegreeOfParallelism = 4 }, i =>
            {
                results[i] = inputs[i].SymmetricEigen();
                svds[i] = inputs[i].Svd();
            });

            for (int i = 0; i < inputs.Length; i++)
            {
                var a = inputs[i];
                var v = results[i].Vectors!;
                AssertRatio(Difference(MultiplyNaive(a, v), MultiplyNaive(v, Matrix<double>.FromDiagonal(results[i].Values))), Norm(a), n, "AV - VΛ");
                var usv = MultiplyNaive(MultiplyNaive(svds[i].U!, Matrix<double>.FromDiagonal(svds[i].Values)), svds[i].V!, transposeB: true);
                AssertRatio(Difference(a, usv), Norm(a), n, "A - UΣVᵀ");
            }
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    /// <summary>最大スレッド数を 1 と全コアにして同じ分解を計算する</summary>
    /// <summary>
    /// 三角方程式（行列の右辺）・LU 分解による求解・コレスキー因子からの逆行列は、分け方が次数だけで決まるので
    /// 単一スレッドと並列でビット単位で同じになる
    /// </summary>
    [Theory]
    [InlineData(130)]
    [InlineData(300)]
    public void SolvesAreIdenticalAcrossThreadCounts(int n)
    {
        var a = Random(n, n, 900 + n);
        for (int i = 0; i < n; i++)
        {
            a[i, i] += n;
        }

        var b = Random(n, 200, 910 + n);
        var (lower1, lowerN) = RunBoth(() => XMath.SolveLowerTriangular(a, b));
        Assert.Equal(lower1.ToArray(), lowerN.ToArray());
        var (upper1, upperN) = RunBoth(() => XMath.SolveUpperTriangular(a.T, b));
        Assert.Equal(upper1.ToArray(), upperN.ToArray());
        var (lu1, luN) = RunBoth(() => XMath.Solve(a, b));
        Assert.Equal(lu1.ToArray(), luN.ToArray());

        var chol = RandomPositiveDefinite(n, 920 + n).Cholesky();
        var (inverse1, inverseN) = RunBoth(() => chol.Inverse());
        Assert.Equal(inverse1.ToArray(), inverseN.ToArray());
        var (solve1, solveN) = RunBoth(() => chol.Solve(b));
        Assert.Equal(solve1.ToArray(), solveN.ToArray());
    }

    /// <summary>
    /// 長いベクトルの縮約は 2 の冪個のブロックの区間ごとに並列に求めるが、和の木は長さだけで決まるので、
    /// 単一スレッドとも、ストライド付きのビュー（逐次のスカラーの経路）ともビット単位で同じになる
    /// </summary>
    [Theory]
    [InlineData(524_287)]
    [InlineData(524_288 + 3 * 64 + 5)]
    [InlineData(1_000_003)]
    [InlineData(2 * 1_048_576 + 1)]
    public void LongVectorReductionsAreIdenticalAcrossThreadCounts(int n)
    {
        var random = new Random(n);
        var x = new Matrix<double>(2, n);
        var y = new Matrix<double>(2, n);
        var xs = new Matrix<float>(2, n);
        for (int i = 0; i < n; i++)
        {
            x[1, i] = (2 * random.NextDouble() - 1) * Math.ScaleB(1, random.Next(-20, 21));
            y[1, i] = 2 * random.NextDouble() - 1;
            xs[1, i] = (float)x[1, i];
        }

        var dx = new Tuna.LinearAlgebra.Vector<double>(x.Row(1).ToArray());
        var dy = new Tuna.LinearAlgebra.Vector<double>(y.Row(1).ToArray());
        var ds = new Tuna.LinearAlgebra.Vector<float>(xs.Row(1).ToArray());
        double[] Reductions(ReadOnlyVectorSpan<double> v, ReadOnlyVectorSpan<double> w, ReadOnlyVectorSpan<float> f) =>
        [
            XMath.Sum(v), XMath.Mean(v), XMath.Dot(v, w), XMath.SumOfSquares(v), XMath.Norm(v), XMath.Norm1(v), XMath.NormInfinity(v),
            XMath.Max(v), XMath.Min(v), XMath.Variance(v, 1), XMath.Sum(f), XMath.Dot(f, f), XMath.Norm(f), XMath.Max(f),
        ];

        var (one, many) = RunBoth(() => Reductions(dx, dy, ds));
        Assert.Equal(one, many);
        Assert.Equal(Reductions(x.Row(1), y.Row(1), xs.Row(1)), many);
    }

    /// <summary>単精度と複素数の行列積も、各要素の積和の順序がスレッド数によらないので、単一スレッドと並列でビット単位で同じになる</summary>
    [Theory]
    [InlineData(300, 260, 500)]
    [InlineData(700, 90, 300)]
    public void SingleAndComplexProductsAreIdenticalAcrossThreadCounts(int m, int n, int k)
    {
        var a = Random(m, k, 950 + m).ConvertChecked<float>();
        var b = Random(k, n, 960 + n).ConvertChecked<float>();
        var (single1, singleN) = RunBoth(() => (XMath.Multiply(a, b), XMath.Multiply(XMath.Transpose(a).T, XMath.Transpose(b).T)));
        Assert.Equal(single1.Item1.ToArray(), singleN.Item1.ToArray());
        Assert.Equal(single1.Item2.ToArray(), singleN.Item2.ToArray());

        var x = ComplexMatrix(Random(m, k, 970 + m), Random(m, k, 980 + m));
        var y = ComplexMatrix(Random(k, n, 990 + n), Random(k, n, 1000 + n));
        var (complex1, complexN) = RunBoth(() => (XMath.Multiply(x, y), XMath.Multiply(XMath.Transpose(x).T, XMath.Transpose(y).T)));
        Assert.Equal(complex1.Item1.ToArray(), complexN.Item1.ToArray());
        Assert.Equal(complex1.Item2.ToArray(), complexN.Item2.ToArray());
    }

    /// <summary>
    /// 行数が数百万の積でも、行のパネルを分担する範囲の境界（パネルの数 × 範囲の番号 / 範囲の数）が溢れない。
    /// float は 600万行、倍精度は 300万行で、AVX-512・AVX2 のどちらのパネルの高さでも積が int を超える（溢れて範囲外を読み書きし、異常終了していた）
    /// </summary>
    [Fact]
    public void TallProductsSplitRowsWithoutOverflow()
    {
        var a = new Matrix<float>(6_000_000, 1);
        a.AsSpan().Fill(1);
        var b = new Matrix<float>(1, 1);
        b[0, 0] = 2;
        var (single1, singleN) = RunBoth(() => XMath.Multiply(a, b));
        Assert.Equal(-1, single1.AsSpan().IndexOfAnyExcept(2f));
        Assert.Equal(-1, singleN.AsSpan().IndexOfAnyExcept(2f));

        var x = new Matrix<double>(3_000_000, 1);
        x.AsSpan().Fill(1);
        var y = new Matrix<double>(1, 1);
        y[0, 0] = 2;
        var (double1, doubleN) = RunBoth(() => XMath.Multiply(x, y));
        Assert.Equal(-1, double1.AsSpan().IndexOfAnyExcept(2.0));
        Assert.Equal(-1, doubleN.AsSpan().IndexOfAnyExcept(2.0));
    }

    private static Matrix<System.Numerics.Complex> ComplexMatrix(Matrix<double> re, Matrix<double> im)
    {
        var z = new Matrix<System.Numerics.Complex>(re.Rows, re.Columns);
        for (int j = 0; j < re.Columns; j++)
        {
            for (int i = 0; i < re.Rows; i++)
            {
                z[i, j] = new System.Numerics.Complex(re[i, j], im[i, j]);
            }
        }

        return z;
    }

    /// <summary>小さな行列のバッチは行列の範囲ごとに並列に計算し、各行列の結果はスレッド数にも1つずつ計算した結果にもよらない</summary>
    [Fact]
    public void SmallTensoresAreIdenticalAcrossThreadCounts()
    {
        int n = 20, count = 300;
        var matrices = new Matrix<double>[count];
        for (int k = 0; k < count; k++)
        {
            matrices[k] = RandomPositiveDefinite(n, 930 + k);
        }

        var a = Tensor<double>.FromMatrices(matrices);
        var rhs = Random(n, count, 940);
        var (one, many) = RunBoth(() =>
        {
            var chol = BatchCholeskyDecomposition.Compute(a);
            return (chol.L, chol.Solve(rhs), chol.Inverse(), XMath.Solve(a, rhs.Unsqueeze(1)));
        });

        Assert.Equal(one.L.ToArray(), many.L.ToArray());
        Assert.Equal(one.Item2.ToArray(), many.Item2.ToArray());
        Assert.Equal(one.Item3.ToArray(), many.Item3.ToArray());
        Assert.Equal(one.Item4.ToArray(), many.Item4.ToArray());
        for (int k = 0; k < count; k += 37)
        {
            Assert.Equal(matrices[k].Cholesky().L.ToArray(), many.L.GetMatrix(k).ToArray());
        }
    }

    private static (T One, T Many) RunBoth<T>(Func<T> compute)
    {
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = 1;
            var one = compute();
            LinearAlgebraSettings.MaxDegreeOfParallelism = Math.Max(2, Environment.ProcessorCount);
            var many = compute();
            return (one, many);
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    private static double[] Diagonal(Matrix<double> a)
    {
        var d = new double[Math.Min(a.Rows, a.Columns)];
        for (int i = 0; i < d.Length; i++)
        {
            d[i] = a[i, i];
        }

        return d;
    }

    /// <summary>値の差が <c>‖A‖·n·ε</c> の閾値倍以内</summary>
    private static void AssertClose(double[] expected, double[] actual, double scale, int n)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            AssertRatio(Math.Abs(expected[i] - actual[i]), scale, n, $"value[{i}]");
        }
    }
}
