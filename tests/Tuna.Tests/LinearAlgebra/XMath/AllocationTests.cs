using Tuna.LinearAlgebra;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>出力先を渡す行列演算が、逐次の呼び出しで必要なもの以外を割り当てないこと</summary>
[Collection(nameof(ParallelismCollection))]
public class AllocationTests
{
    public static TheoryData<string> SequentialOperations => new()
    {
        "Multiply", "SolveLowerTriangular", "CholeskySolve",
    };

    [Theory, MemberData(nameof(SequentialOperations))]
    public void SequentialMatrixOperationsDoNotAllocate(string operation)
    {
        WithThreads(1, () =>
        {
            // 行列積はパッキングする大きさ、分解と三角方程式は再帰的に分ける大きさにする
            const int n = 96;
            var a = PositiveDefinite(n);
            var b = General(n, 3);
            var destination = new Matrix<double>(n, n);
            var l = CholeskyDecomposition.Compute(a).L;
            var cholesky = CholeskyDecomposition.Compute(a);
            Action call = operation switch
            {
                "Multiply" => () => XMath.Multiply<double>(a.AsMatrixSpan(), b.AsMatrixSpan(), destination.AsMatrixSpan()),
                "SolveLowerTriangular" => () => XMath.SolveLowerTriangular<double>(l.AsMatrixSpan(), b.AsMatrixSpan(), destination.AsMatrixSpan()),
                "CholeskySolve" => () => cholesky.Solve(b.AsMatrixSpan(), destination.AsMatrixSpan()),
                _ => throw new ArgumentOutOfRangeException(nameof(operation)),
            };

            Assert.Equal(0L, MinimumAllocation(call));
        });
    }

    [Fact]
    public void CholeskyComputeWithDestinationAllocatesOnlyTheDecomposition()
    {
        WithThreads(1, () =>
        {
            // 返す CholeskyDecomposition のオブジェクトだけで、再帰の各段で行列積の状態やクロージャを確保しない
            const int n = 96;
            var a = PositiveDefinite(n);
            var destination = new Matrix<double>(n, n);
            long allocated = MinimumAllocation(() => CholeskyDecomposition.Compute(a, destination));
            Assert.InRange(allocated / 64, 0, 64);
        });
    }

    [Theory]
    [InlineData(65)]
    [InlineData(300)]
    public void CholeskyInverseAllocatesOnlyTheResult(int n)
    {
        WithThreads(1, () =>
        {
            // 作業領域は n によらない小さいもの（プールから借りる）だけで、n × n の作業領域は確保しない
            var cholesky = CholeskyDecomposition.Compute(PositiveDefinite(n));
            long allocated = MinimumAllocation(() => cholesky.Inverse(), calls: 4);
            Assert.InRange(allocated / 4, (long)n * n * sizeof(double), (long)n * n * sizeof(double) + 1024);
        });
    }

    [Fact]
    public void InverseWithDestinationAllocatesOnlyTheFactorization()
    {
        WithThreads(1, () =>
        {
            // LU 分解の写しとピボットだけで、単位行列は別に作らない
            const int n = 200;
            var a = General(n, 5);
            var destination = new Matrix<double>(n, n);
            long allocated = MinimumAllocation(() => XMath.Inverse(a.AsMatrixSpan(), destination.AsMatrixSpan()), calls: 4);
            Assert.InRange(allocated / 4, (long)n * n * sizeof(double), (long)n * n * sizeof(double) + n * sizeof(int) + 1024);
        });
    }

    /// <summary>
    /// 呼び出しを何回か繰り返したときの割り当ての最小値。段階的 JIT の切り替えの途中で、実行時が一度だけ数 KB を割り当てることがあるので、
    /// 何回か測って最小の回を使う（呼び出しごとの割り当てならどの回にも現れる）
    /// </summary>
    private static long MinimumAllocation(Action call, int calls = 64)
    {
        for (int i = 0; i < 64; i++)
        {
            call();
        }

        long minimum = long.MaxValue;
        for (int round = 0; round < 5; round++)
        {
            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < calls; i++)
            {
                call();
            }

            minimum = Math.Min(minimum, GC.GetAllocatedBytesForCurrentThread() - before);
        }

        return minimum;
    }

    private static Matrix<double> PositiveDefinite(int n)
    {
        var a = new Matrix<double>(n, n);
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                a[i, j] = i == j ? 2.0 + (double)i / n : 0.5 / (1 + Math.Abs(i - j));
            }
        }

        return a;
    }

    private static Matrix<double> General(int n, int seed)
    {
        var random = new Random(seed);
        var a = new Matrix<double>(n, n);
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                a[i, j] = random.NextDouble() - 0.5 + (i == j ? n * 0.05 : 0);
            }
        }

        return a;
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
