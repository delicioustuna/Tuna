using System.Numerics;
using System.Reflection;
using Tuna.LinearAlgebra;
using static Tuna.Tests.LinearAlgebra.LinearAlgebraTestUtilities;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>並列カーネルの状態と作業デリゲートを、呼び出しのたびに確保しないこと</summary>
[Collection(nameof(ParallelismCollection))]
public class ParallelAllocationTests
{
    private delegate int Factor(int n, Span<double> a, int lda);

    public static TheoryData<int, int> ReductionSizesAndThreads
    {
        get
        {
            var data = new TheoryData<int, int>();
            foreach (int threads in new[] { 1, 2, 3, 4 })
            {
                foreach (int length in new[] { 524287, 524288, 1000000, 3000001 })
                {
                    data.Add(threads, length);
                }
            }

            return data;
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void MatrixProductsWithDestinationsDoNotAllocate(int threads)
    {
        WithThreads(threads, () =>
        {
            MatrixProducts<float>();
            MatrixProducts<double>();
        });
    }

    private static void MatrixProducts<T>()
        where T : unmanaged, INumberBase<T>
    {
        // 内積方向のブロック境界とマイクロタイルの端をまたぎ、B のパックも並列にする。
        const int m = 257, k = 263, n = 259;
        foreach (bool transA in new[] { false, true })
        {
            foreach (bool transB in new[] { false, true })
            {
                var a = new Matrix<T>(transA ? k : m, transA ? m : k);
                var b = new Matrix<T>(transB ? n : k, transB ? k : n);
                var destination = new Matrix<T>(m, n);
                a.AsSpan().Fill(T.One);
                b.AsSpan().Fill(T.One);
                AssertNoAllocation(() => XMath.Multiply<T>(
                    transA ? a.T : a.AsMatrixSpan(), transB ? b.T : b.AsMatrixSpan(), destination));
                foreach (T value in destination.AsSpan())
                {
                    Assert.Equal(T.CreateChecked(k), value);
                }
            }
        }
    }

    [Theory]
    [MemberData(nameof(ReductionSizesAndThreads))]
    public void LongReductionsAndSoftmaxDoNotAllocate(int threads, int length)
    {
        WithThreads(threads, () =>
        {
            Reductions<float>(length);
            Reductions<double>(length);
        });
    }

    private static void Reductions<T>(int length)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var x = new Tuna.LinearAlgebra.Vector<T>(length);
        var y = new Tuna.LinearAlgebra.Vector<T>(length);
        var destination = new Tuna.LinearAlgebra.Vector<T>(length);
        x.AsSpan().Fill(T.CreateChecked(0.25));
        y.AsSpan().Fill(T.CreateChecked(-0.5));
        T result = T.Zero;
        AssertNoAllocation(() => result = XMath.Sum<T>(x));
        Assert.Equal(T.CreateChecked(length) / T.CreateChecked(4), result);
        AssertNoAllocation(() => result = XMath.Dot<T>(x, y));
        Assert.Equal(-T.CreateChecked(length) / T.CreateChecked(8), result);
        AssertNoAllocation(() => result = XMath.Max<T>(x));
        Assert.Equal(T.CreateChecked(0.25), result);
        AssertNoAllocation(() => result = XMath.LogSumExp<T>(x));
        Assert.True(T.IsFinite(result));
        AssertNoAllocation(() => XMath.LogSoftmax<T>(x, destination));
        T expectedLog = T.CreateTruncating(-Math.Log(length));
        Assert.InRange(destination[0], T.BitDecrement(expectedLog), T.BitIncrement(expectedLog));
        Assert.Equal(destination[0], destination[length - 1]);
        AssertNoAllocation(() => XMath.Softmax<T>(x, destination));
        T expected = T.One / T.CreateChecked(length);
        Assert.Equal(expected, destination[0]);
        Assert.Equal(expected, destination[length - 1]);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void MatrixVectorProductsWithDestinationsDoNotAllocate(int threads)
    {
        WithThreads(threads, () =>
        {
            MatrixVectorProducts<float>(513, 521);
            MatrixVectorProducts<double>(513, 521);
            // 1列が 256 KiB を超えると、部分和の列分割ではなく行分割を通る。
            MatrixVectorProducts<float>(65537, 8);
            MatrixVectorProducts<double>(32769, 8);
        });
    }

    private static void MatrixVectorProducts<T>(int m, int n)
        where T : unmanaged, INumberBase<T>
    {
        var a = new Matrix<T>(m, n);
        a.AsSpan().Fill(T.One);
        foreach (bool transpose in new[] { false, true })
        {
            int columns = transpose ? m : n;
            var x = new Tuna.LinearAlgebra.Vector<T>(columns);
            x.AsSpan().Fill(T.One);
            var destination = new Tuna.LinearAlgebra.Vector<T>(transpose ? n : m);
            AssertNoAllocation(() => XMath.Multiply<T>(transpose ? a.T : a.AsMatrixSpan(), x, destination));
            foreach (T value in destination.AsSpan())
            {
                Assert.Equal(T.CreateChecked(columns), value);
            }
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void OtherLevel2ParallelPathsDoNotAllocate(int threads)
    {
        WithThreads(threads, () =>
        {
            const int n = 1153, lda = n + 3;
            var a = new double[lda * n];
            var x = new double[n];
            var y = new double[n];
            Array.Fill(x, 1.0);
            for (int i = 0; i < n; i++)
            {
                a[i + i * lda] = 1.0;
            }

            AssertNoAllocation(() => Blas.Ger(n, n, 0.125, x, x, a, lda));
            // GER の繰り返しで変えた行列を、単位行列へ戻す。
            Array.Clear(a);
            for (int i = 0; i < n; i++)
            {
                a[i + i * lda] = 1.0;
            }

            foreach (BlasTriangle triangle in Enum.GetValues<BlasTriangle>())
            {
                AssertNoAllocation(() => Blas.Symv(triangle, n, 1.0, a, lda, x, 0.0, y));
                Assert.Equal(x, y);
                foreach (BlasTranspose transpose in new[] { BlasTranspose.None, BlasTranspose.Transpose })
                {
                    AssertNoAllocation(() => Blas.Trmv(triangle, transpose, BlasDiagonal.NonUnit, n, a, lda, x));
                    AssertNoAllocation(() => Blas.Trsv(triangle, transpose, BlasDiagonal.NonUnit, n, a, lda, x));
                    Assert.All(x, value => Assert.Equal(1.0, value));
                }
            }

            // α = 0 の GEMM は、列間の余りを保って β·C だけを計算する。
            var b = new double[n];
            foreach (double beta in new[] { 0.0, 0.5 })
            {
                AssertNoAllocation(() => Blas.Gemm(BlasTranspose.None, BlasTranspose.None,
                    n, n, 1, 0.0, x, n, b, 1, beta, a, lda));
            }
        });
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void ComplexLevel2ParallelPathsDoNotAllocate(int threads)
    {
        WithThreads(threads, () =>
        {
            const int n = 259, lda = n + 5;
            var a = new Complex[lda * n];
            var x = new Complex[n];
            var y = new Complex[n];
            Array.Fill(x, new Complex(0.5, -0.25));
            for (int i = 0; i < n; i++)
            {
                a[i + i * lda] = Complex.One;
            }

            foreach (BlasTranspose transpose in Enum.GetValues<BlasTranspose>())
            {
                AssertNoAllocation(() => Blas.Gemv(transpose, n, n, Complex.One, a, lda, x, Complex.Zero, y));
                Assert.Equal(x, y);
            }

            foreach (BlasTriangle triangle in Enum.GetValues<BlasTriangle>())
            {
                AssertNoAllocation(() => Blas.Hemv(triangle, n, Complex.One, a, lda, x, Complex.Zero, y));
                Assert.Equal(x, y);
            }

            AssertNoAllocation(() => Blas.GerConjugate(n, n, new Complex(0.125, 0.25), x, x, a, lda));
        });
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void CholeskyAndTriangularSolvesDoNotAllocate(int threads)
    {
        WithThreads(threads, () =>
        {
            // 128 次以上の分解は右視的ブロック版で、パネルの三角方程式と後続の更新を並列にする。
            // 逆行列は 64 列のブロックの三角方程式と、行のブロックごとの行列積を並列にする（300 次で 5 ブロック）。
            // 右辺が 128 列以上の三角方程式は、列の範囲ごとに並列に解く（257 列は 4 本に分けると端数が残る）。
            const int n = 300, nrhs = 257;
            var a = RandomPositiveDefinite(n, 960);
            var b = Random(n, nrhs, 961);
            var factor = typeof(Matrix<double>).Assembly.GetType("Tuna.LinearAlgebra.CholeskyKernels")!
                .GetMethod("Factor", BindingFlags.Static | BindingFlags.NonPublic)!.CreateDelegate<Factor>();
            var cholesky = CholeskyDecomposition.Compute(a);
            var l = cholesky.L;
            var factored = new double[n * n];
            var inverse = new Matrix<double>(n, n);
            var solved = new Matrix<double>(n, nrhs);
            var lower = new Matrix<double>(n, nrhs);
            var upper = new Matrix<double>(n, nrhs);
            int info = -1;

            AssertNoAllocation(() =>
            {
                a.AsSpan().CopyTo(factored);
                info = factor(n, factored, n);
            });
            Assert.Equal(0, info);
            for (int j = 0; j < n; j++)
            {
                Assert.True(factored.AsSpan(j + j * n, n - j).SequenceEqual(l.AsSpan().Slice(j + j * n, n - j)));
            }

            AssertNoAllocation(() => cholesky.Inverse(inverse.AsMatrixSpan()));
            Assert.Equal(cholesky.Inverse().ToArray(), inverse.ToArray());
            AssertNoAllocation(() => cholesky.Solve(b.AsMatrixSpan(), solved.AsMatrixSpan()));
            Assert.Equal(cholesky.Solve(b.AsMatrixSpan()).ToArray(), solved.ToArray());
            AssertNoAllocation(() => XMath.SolveLowerTriangular<double>(l.AsMatrixSpan(), b.AsMatrixSpan(), lower.AsMatrixSpan()));
            Assert.Equal(XMath.SolveLowerTriangular(l, b).ToArray(), lower.ToArray());
            AssertNoAllocation(() => XMath.SolveUpperTriangular<double>(l.T, b.AsMatrixSpan(), upper.AsMatrixSpan()));
            Assert.Equal(XMath.SolveUpperTriangular(l.T, b).ToArray(), upper.ToArray());
        });
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void RecomputedDecompositionsDoNotAllocate(int threads)
    {
        WithThreads(threads, () =>
        {
            // 300 次の LU は再帰の右側で行の交換・三角方程式・行列積を並列にし、Cholesky は右視的ブロック版を通る。
            const int n = 300, nrhs = 257;
            var spd = RandomPositiveDefinite(n, 970);
            var general = Random(n, n, 971);
            var b = Random(n, nrhs, 972);
            var cholesky = CholeskyDecomposition.Compute(RandomPositiveDefinite(n, 973));
            var lu = Random(n, n, 974).Lu();
            var solved = new Matrix<double>(n, nrhs);
            var transposed = new Matrix<double>(n, nrhs);
            var inverse = new Matrix<double>(n, n);
            var vector = new Tuna.LinearAlgebra.Vector<double>(n);
            double logDeterminant = 0;

            AssertNoAllocation(() =>
            {
                cholesky.Recompute(spd);
                cholesky.Solve(b.AsMatrixSpan(), solved.AsMatrixSpan());
                logDeterminant = cholesky.LogDeterminant;
            });
            var fresh = CholeskyDecomposition.Compute(spd);
            Assert.Equal(fresh.L.ToArray(), cholesky.L.ToArray());
            Assert.Equal(fresh.Solve(b).ToArray(), solved.ToArray());
            Assert.Equal(fresh.LogDeterminant, logDeterminant);

            AssertNoAllocation(() =>
            {
                lu.Recompute(general);
                lu.Solve(b.AsMatrixSpan(), solved.AsMatrixSpan());
                lu.SolveTransposed(b.AsMatrixSpan(), transposed.AsMatrixSpan());
                lu.SolveTransposed(b.Column(0), vector.AsVectorSpan());
                lu.Inverse(inverse.AsMatrixSpan());
                logDeterminant = lu.LogAbsDeterminant;
            });
            var freshLu = general.Lu();
            Assert.Equal(freshLu.U.ToArray(), lu.U.ToArray());
            Assert.Equal(freshLu.Solve(b).ToArray(), solved.ToArray());
            Assert.Equal(freshLu.SolveTransposed(b).ToArray(), transposed.ToArray());
            Assert.Equal(freshLu.SolveTransposed(b.Column(0)).ToArray(), vector.ToArray());
            Assert.Equal(freshLu.Inverse().ToArray(), inverse.ToArray());
            Assert.Equal(freshLu.LogAbsDeterminant, logDeterminant);
        });
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void RecomputedOrthogonalDecompositionsDoNotAllocate(int threads)
    {
        WithThreads(threads, () =>
        {
            // 300 次の QR はパネルの分解と更新を重ねる並列版と、Q を作るブロックの並列版を通る。
            // 対称固有値分解・特異値分解は列の分担を固定した縮約と並列の分割統治法を通る。
            // 特異値分解の縦長は先に QR 分解し、横長は転置して解く。
            const int n = 300;
            var general = Random(n, n, 980);
            var symmetric = RandomSymmetric(n, 981);
            var tall = Random(n, n / 2, 982);
            var qr = Random(n, n, 983).Qr();
            var eig = RandomSymmetric(n, 984).SymmetricEigen();
            var svd = Random(n, n, 985).Svd();
            var tallSvd = Random(n, n / 2, 986).Svd();
            var wideSvd = Random(n / 2, n, 987).Svd();

            AssertNoAllocation(() => qr.Recompute(general));
            var freshQr = general.Qr();
            Assert.Equal(freshQr.Q.ToArray(), qr.Q.ToArray());
            Assert.Equal(freshQr.R.ToArray(), qr.R.ToArray());

            AssertNoAllocation(() => eig.Recompute(symmetric));
            var freshEig = symmetric.SymmetricEigen();
            Assert.Equal(freshEig.Values, eig.Values);
            Assert.Equal(freshEig.Vectors!.ToArray(), eig.Vectors!.ToArray());

            AssertNoAllocation(() =>
            {
                svd.Recompute(general);
                tallSvd.Recompute(tall);
                wideSvd.Recompute(tall.T);
            });
            var freshSvd = general.Svd();
            Assert.Equal(freshSvd.Values, svd.Values);
            Assert.Equal(freshSvd.U!.ToArray(), svd.U!.ToArray());
            Assert.Equal(freshSvd.V!.ToArray(), svd.V!.ToArray());
            Assert.Equal(tall.Svd().U!.ToArray(), tallSvd.U!.ToArray());
            Assert.Equal(tall.T.ToMatrix().Svd().V!.ToArray(), wideSvd.V!.ToArray());
        });
    }

    // 並列の作業の中（入れ子で逐次に実行する行列積も含む）で借りる配列は、作業を実行するスロットごとの置き場から出すので、
    // どのスレッドがどの作業を受け持っても、同じ分解し直しを2回した後は置き場が足りなくならず、確保しない。共有の ArrayPool から借りていたときは、
    // 並列数 4 の 300 × 200 の特異値分解・300 次の対称固有値分解を 300 回くり返すと、1 プロセスでおよそ1回、16 KB などの配列を確保していた。
    // gen2 の GC の後に置き場を片付けても（使っている配列は手放さない）足りなくならない。
    [Fact]
    public void RepeatedParallelRecomputeNeverRunsShortOfWorkArrays()
    {
        WithThreads(4, () =>
        {
            var general = Random(300, 200, 990);
            var square = Random(300, 300, 991);
            var symmetric = RandomSymmetric(300, 992);
            var qr = general.Qr();
            var lu = square.Lu();
            var svd = general.Svd();
            var eig = symmetric.SymmetricEigen();
            AssertNoShortageInAnyCall(() => qr.Recompute(general), 100);
            AssertNoShortageInAnyCall(() => lu.Recompute(square), 100);
            AssertNoShortageInAnyCall(() => svd.Recompute(general), 150);
            AssertNoShortageInAnyCall(() => eig.Recompute(symmetric), 150);
        });
    }

    // 区間の後にほかのスロットのために先に確保するのは、すべての型・スロットを合わせて 16 MiB まで（小さい区分から足す）。
    // 上限がないと、作業が1つの要素で 16 MiB の配列を借りただけで、区間の後にほかのスロットにも 16 MiB ずつ確保し、
    // GC ヒープに上限があると、計算が終わった後のそこで OutOfMemoryException になった（#136 のレビュー）。
    [Fact]
    public void LargeWorkArraysAreNotPreallocatedForOtherSlots()
    {
        WithThreads(4, () =>
        {
            long allocated = AllocatedByRentingOnceInWork(1 << 21);
            Assert.True(allocated < (1 << 21) * sizeof(double) + ResultBytes + FillLimitBytes + (1 << 20), $"allocated {allocated} B");
        });
    }

    // 上限は1スロットごとではなく、すべてのスロットを合わせたもの（1スロットに型ごとに 4 MiB の上限では、並列数 32 で 2 MiB の配列を 62 MiB 先に確保した。#136 のレビュー）。
    // 8 MiB の配列は、上限がなければほかの3スロットに 24 MiB 先に確保し、上限の中では2スロットまで。
    [Fact]
    public void PreallocatedWorkArraysForAllSlotsStayWithinOneLimit()
    {
        WithThreads(4, () =>
        {
            long allocated = AllocatedByRentingOnceInWork(1 << 20);
            Assert.True(allocated < (1 << 20) * sizeof(double) + ResultBytes + FillLimitBytes + (1 << 20), $"allocated {allocated} B");
        });
    }

    private const long ResultBytes = 512 * 512 * sizeof(double), FillLimitBytes = 16L << 20;

    // Generate の作業の (0, 0) だけが、作業の中で長さ length の配列を1つ借りて返す（共有のプールになければ確保する）。区間の後に足すのは、ほかの区分を合わせても上限まで。
    // その呼び出しで確保したバイト数（プロセス全体）を返す。
    private static long AllocatedByRentingOnceInWork(int length)
    {
        var pool = typeof(Matrix<double>).Assembly.GetType("Tuna.LinearAlgebra.WorkArrayPool`1")!.MakeGenericType(typeof(double));
        var rent = pool.GetMethod("Rent", BindingFlags.Static | BindingFlags.NonPublic)!.CreateDelegate<Func<int, double[]>>();
        var giveBack = pool.GetMethod("Return", BindingFlags.Static | BindingFlags.NonPublic)!.CreateDelegate<Action<double[], bool>>();
        Func<int, int, double> op = (i, j) =>
        {
            if (i == 0 && j == 0)
            {
                giveBack(rent(length), false);
            }

            return i + j;
        };

        long before = GC.GetTotalAllocatedBytes(precise: true);
        var result = Matrix<double>.Generate(512, 512, op);
        long allocated = GC.GetTotalAllocatedBytes(precise: true) - before;
        Assert.Equal(1022, result[511, 511]);
        return allocated;
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void ParallelismWithAnExistingActionDoesNotAllocate(int threads)
    {
        WithThreads(threads, () =>
        {
            var run = GetFor();
            var visits = new int[13];
            Action<int> body = index => Interlocked.Increment(ref visits[index]);
            AssertNoAllocation(() => run(visits.Length, threads, body));
            Assert.All(visits, value => Assert.Equal(64, value));

            var type = typeof(Matrix<double>).Assembly.GetType("Tuna.LinearAlgebra.Parallelism")!;
            var runStatic = type.GetMethods(BindingFlags.Static | BindingFlags.NonPublic)
                .Single(method => method.Name == "ForStatic" && !method.IsGenericMethod)
                .CreateDelegate<Action<int, Action<int>>>();
            Array.Clear(visits);
            AssertNoAllocation(() => runStatic(threads, body));
            for (int i = 0; i < visits.Length; i++)
            {
                Assert.Equal(i < threads ? 64 : 0, visits[i]);
            }
        });
    }

    [Fact]
    public void NestedAndConcurrentProductsKeepTheirOwnState()
    {
        WithThreads(4, () =>
        {
            const int n = 263;
            var a = new Matrix<double>(n, n);
            var x = new Tuna.LinearAlgebra.Vector<double>(n);
            for (int j = 0; j < n; j++)
            {
                x[j] = (j % 7 - 3) * 0.125;
                for (int i = 0; i < n; i++)
                {
                    a[i, j] = (i % 11 - j % 5) * 0.25;
                }
            }

            var expected = XMath.Multiply(a, x).ToArray();
            var destinations = Enumerable.Range(0, 8).Select(_ => new Tuna.LinearAlgebra.Vector<double>(n)).ToArray();
            Action<int> multiply = index => XMath.Multiply<double>(a, x, destinations[index]);
            // 外側の並列区間の busy を見て、内側の状態付き For は呼んだスレッドで順に実行する。
            GetFor()(destinations.Length, 4, multiply);
            Parallel.For(0, destinations.Length, multiply);
            foreach (var destination in destinations)
            {
                Assert.Equal(expected, destination.ToArray());
            }

            var expectedProduct = XMath.Multiply(a, a).ToArray();
            var products = Enumerable.Range(0, 8).Select(_ => new Matrix<double>(n, n)).ToArray();
            Parallel.For(0, products.Length, index => XMath.Multiply<double>(a, a, products[index]));
            foreach (var product in products)
            {
                Assert.Equal(expectedProduct, product.ToArray());
            }
        });
    }

    [Fact]
    public void InterruptedCallerWaitsForWorkersBeforeReturning()
    {
        WithThreads(4, () =>
        {
            if (Environment.ProcessorCount < 2)
            {
                return;
            }

            var run = GetFor();
            using var workerEntered = new ManualResetEventSlim();
            using var callerBodyFinished = new ManualResetEventSlim();
            using var releaseWorker = new ManualResetEventSlim();
            Exception? error = null;
            int callerId = 0;
            Action<int> body = _ =>
            {
                if (Environment.CurrentManagedThreadId != callerId)
                {
                    workerEntered.Set();
                    releaseWorker.Wait();
                }
                else
                {
                    workerEntered.Wait();
                    callerBodyFinished.Set();
                }
            };
            var caller = new Thread(() =>
            {
                callerId = Environment.CurrentManagedThreadId;
                try
                {
                    run(2, 2, body);
                }
                catch (Exception ex)
                {
                    error = ex;
                }
            });
            caller.Start();
            try
            {
                Assert.True(callerBodyFinished.Wait(TimeSpan.FromSeconds(10)));
                // 呼び出し元が作業を終え、合流の SpinWait で待機するまで待つ。
                Assert.True(SpinWait.SpinUntil(
                    () => (caller.ThreadState & ThreadState.WaitSleepJoin) != 0, TimeSpan.FromSeconds(10)));
                caller.Interrupt();
                Assert.False(caller.Join(TimeSpan.FromMilliseconds(100)));
            }
            finally
            {
                releaseWorker.Set();
                Assert.True(caller.Join(TimeSpan.FromSeconds(10)));
            }

            Assert.IsType<ThreadInterruptedException>(error);
            int visits = 0;
            run(13, 4, _ => Interlocked.Increment(ref visits));
            Assert.Equal(13, visits);
        });
    }

    private static Action<int, int, Action<int>> GetFor()
    {
        var type = typeof(Matrix<double>).Assembly.GetType("Tuna.LinearAlgebra.Parallelism")!;
        return type.GetMethods(BindingFlags.Static | BindingFlags.NonPublic)
            .Single(method => method.Name == "For" && !method.IsGenericMethod)
            .CreateDelegate<Action<int, int, Action<int>>>();
    }

    internal static void AssertNoAllocation(Action call)
    {
        for (int i = 0; i < 32; i++)
        {
            call();
        }

        // 呼び出しごとの確保なら、どの窓にも現れる。作業の中で共有の ArrayPool から借りていたときは、並列の作業の割り当てが変わると
        // 窓の中で一度だけ確保が入ることがあった（4 スレッドで数回に1回、2,048〜8,192 要素の配列1つ）。作業の中の配列は
        // スロットごとの置き場から出すようにしてこれは起きなくなった（RepeatedParallelRecomputeNeverRunsShortOfWorkArrays）が、
        // 段階的コンパイルが最適化し直すときの一度きりの確保もあるので、窓は3回まで測り、どれかが 0 なら確保しないとみなす。
        long allocated = 0;
        for (int attempt = 0; attempt < 3; attempt++)
        {
            // 測るスレッドの使いかけの割り当て領域を空にしてから数える。空にしないと、
            // 窓の中で確保しなくても、その残り（8 KB 未満）が増分に入ることがある。
            GC.Collect(0, GCCollectionMode.Forced, blocking: true);
            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 32; i++)
            {
                call();
            }

            allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            if (allocated == 0)
            {
                return;
            }
        }

        Assert.Equal(0L, allocated);
    }

    // 2回呼んで準備し、gen2 の GC で置き場の片付けを起こしてから、1回ずつ、並列の作業の中で置き場が足りなかった回と、
    // 呼んだスレッドが確保した回（数える直前に gen0 の GC を入れる）を数える。置き場が足りなかった回は 0 でなければならない。
    // 確保は、段階的コンパイルが最適化し直すときの一度きりの確保（12 KB ほど。置き場とは関係ない）がまれに入るので、2回以上で失敗とする。
    private static void AssertNoShortageInAnyCall(Action call, int calls)
    {
        var shortages = typeof(Matrix<double>).Assembly.GetType("Tuna.LinearAlgebra.WorkArrayPools")!
            .GetProperty("Shortages", BindingFlags.Static | BindingFlags.NonPublic)!.GetMethod!.CreateDelegate<Func<long>>();
        call();
        call();
        GC.Collect();
        GC.WaitForPendingFinalizers();
        var shortCalls = new List<int>();
        var allocated = new List<string>();
        for (int i = 0; i < calls; i++)
        {
            GC.Collect(0, GCCollectionMode.Forced, blocking: true);
            long count = shortages();
            long before = GC.GetAllocatedBytesForCurrentThread();
            call();
            long bytes = GC.GetAllocatedBytesForCurrentThread() - before;
            if (shortages() != count)
            {
                shortCalls.Add(i + 3);
            }

            if (bytes != 0)
            {
                allocated.Add($"call {i + 3}: {bytes} B");
            }
        }

        Assert.True(shortCalls.Count == 0, $"short of work arrays in calls {string.Join(", ", shortCalls)}");
        Assert.True(allocated.Count <= 1, string.Join(", ", allocated));
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
}
