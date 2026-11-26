using System.Numerics;
using Tuna.LinearAlgebra;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>ベクトルの集約と更新は、並列に実行しても呼び出しのたびに確保しない</summary>
[Trait("LinearAlgebra", "Parallelism")]
[Collection(nameof(ParallelismCollection))]
public class Level1AllocationTests
{
    /// <summary>0 は設定を変えず既定の並列数で確かめる。端数を含み、全型・全更新の並列化の閾値を超える長さを使う</summary>
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(4)]
    public void UpdatesDoNotAllocate(int threads)
    {
        const int length = 300_007;
        var x = new double[length];
        var y = new double[length];
        Array.Fill(x, 0.25);
        Array.Fill(y, -0.5);
        float[] xs = Array.ConvertAll(x, v => (float)v);
        float[] ys = Array.ConvertAll(y, v => (float)v);
        Complex[] xc = Array.ConvertAll(x, v => new Complex(v, v));
        Complex[] yc = Array.ConvertAll(y, v => new Complex(v, -v));
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            if (threads > 0)
            {
                LinearAlgebraSettings.MaxDegreeOfParallelism = threads;
            }

            AssertNoAllocation(() => Blas.Axpy(0.25, x, y));
            AssertNoAllocation(() => Blas.Scale(-1.0, x));
            AssertNoAllocation(() => Blas.Rotate(x, y, 0.0, 1.0));
            AssertNoAllocation(() => Blas.Swap(x, y));
            AssertNoAllocation(() => Blas.Copy(x, y));
            AssertNoAllocation(() => Blas.Axpy(0.25f, xs, ys));
            AssertNoAllocation(() => Blas.Scale(-1f, xs));
            AssertNoAllocation(() => Blas.Rotate(xs, ys, 0f, 1f));
            AssertNoAllocation(() => Blas.Swap(xs, ys));
            AssertNoAllocation(() => Blas.Copy(xs, ys));
            AssertNoAllocation(() => Blas.Axpy(new Complex(0.25, -0.5), xc, yc));
            AssertNoAllocation(() => Blas.Scale(new Complex(0, 1), xc));
            AssertNoAllocation(() => Blas.Scale(-1.0, xc));
            AssertNoAllocation(() => Blas.Rotate(xc, yc, 0.0, 1.0));
            AssertNoAllocation(() => Blas.Swap(xc, yc));
            AssertNoAllocation(() => Blas.Copy(xc, yc));
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    /// <summary>区間が数十個の長さと、区間の数が上限（768 個）に達する長さ</summary>
    [Theory]
    [InlineData(1, 300_007)]
    [InlineData(4, 300_007)]
    [InlineData(1, 4_400_003)]
    [InlineData(4, 4_400_003)]
    public void ReductionsDoNotAllocate(int threads, int length)
    {
        var x = new double[length];
        var y = new double[length];
        Array.Fill(x, 0.25);
        Array.Fill(y, -0.5);
        float[] xs = Array.ConvertAll(x, v => (float)v);
        float[] ys = Array.ConvertAll(y, v => (float)v);
        Complex[] xc = Array.ConvertAll(x, v => new Complex(v, v));
        Complex[] yc = Array.ConvertAll(y, v => new Complex(v, 0));
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = threads;
            double result = 0;
            AssertNoAllocation(() => result = Blas.Dot(x, y));
            Assert.Equal(-length / 8.0, result);
            AssertNoAllocation(() => result = Blas.Norm2(x));
            Assert.Equal(Math.Sqrt(length / 16.0), result);
            AssertNoAllocation(() => result = Blas.SumAbs(x));
            Assert.Equal(length / 4.0, result);
            AssertNoAllocation(() => result = Blas.Dot(xs, ys));
            Assert.Equal(-length / 8.0, result);
            AssertNoAllocation(() => result = Blas.Norm2(xs));
            Assert.Equal((float)Math.Sqrt(length / 16.0), (float)result);
            AssertNoAllocation(() => result = Blas.SumAbs(xs));
            Assert.Equal(length / 4.0, result);
            Complex complex = default;
            AssertNoAllocation(() => complex = Blas.DotConjugate(xc, yc));
            Assert.Equal(new Complex(-length / 8.0, length / 8.0), complex);
            AssertNoAllocation(() => result = Blas.Norm2(xc));
            Assert.Equal(Math.Sqrt(length / 8.0), result);
            AssertNoAllocation(() => result = Blas.SumAbs(xc));
            Assert.Equal(length / 2.0, result);
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    /// <summary>32 回の準備のあと、32 回の呼び出しで呼んだスレッドの確保量が増えないことを確かめる</summary>
    private static void AssertNoAllocation(Action call)
    {
        for (int i = 0; i < 32; i++)
        {
            call();
        }

        // 呼んだスレッドの使いかけの割り当て領域を空にしてから数える（残りが増分に入らないようにする）
        GC.Collect(0, GCCollectionMode.Forced, blocking: true);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 32; i++)
        {
            call();
        }

        Assert.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - before);
    }
}
