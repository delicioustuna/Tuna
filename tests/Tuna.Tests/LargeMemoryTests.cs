using System.Numerics;
using System.Runtime.InteropServices;
using Tuna.LinearAlgebra;
using Tuna.RandomSource;
using Tuna.Tests.LinearAlgebra;

namespace Tuna.Tests;

/// <summary>
/// 要素数が int の上限に近い入力（8〜16 GiB）で、長さから求める個数が溢れないこと。メモリを多く使うので、
/// 環境変数 <c>TUNA_LARGE_MEMORY_TESTS=1</c> のときだけ確かめる（ほかでは何もせずに通る）。
/// <c>TUNA_LARGE_MEMORY_TESTS=1 dotnet test tests/Tuna.Tests/Tuna.Tests.csproj -c Release --filter Category=LargeMemory</c>
/// </summary>
[Trait("Category", "LargeMemory")]
[Collection(nameof(ParallelismCollection))]
public unsafe class LargeMemoryTests
{
    private static bool Enabled => Environment.GetEnvironmentVariable("TUNA_LARGE_MEMORY_TESTS") == "1";

    /// <summary>
    /// 約 21.5 億要素の float の縮約は、並列でも1スレッドと同じ値になる（並列の区間の数の計算が溢れ、2スレッド以上で配列の確保に失敗していた）。
    /// 0 で埋めた領域に少数の値を置く（Linux では、書かない領域は読んでも物理メモリを使わない）
    /// </summary>
    [Fact]
    public void ReductionsOfLongestFloatVectorWorkInParallel()
    {
        if (!Enabled)
        {
            return;
        }

        int n = Array.MaxLength;
        float* p = (float*)NativeMemory.AllocZeroed((nuint)n, sizeof(float));
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            var data = new Span<float>(p, n);
            data[5] = -2;
            data[123456789] = 1.5f;
            data[n - 1] = 3;
            nint address = (nint)p;
            float[] Reductions()
            {
                var x = new ReadOnlyVectorSpan<float>(new ReadOnlySpan<float>((void*)address, n));
                return [XMath.Max(x), XMath.Min(x), XMath.NormInfinity(x), XMath.Norm(x), XMath.Sum(x), XMath.Dot(x, x)];
            }

            LinearAlgebraSettings.MaxDegreeOfParallelism = 1;
            var one = Reductions();
            LinearAlgebraSettings.MaxDegreeOfParallelism = Math.Max(2, Environment.ProcessorCount);
            var many = Reductions();
            Assert.Equal([3f, -2f, 3f], one[..3]);
            Assert.Equal(MathF.Sqrt(15.25f), one[3], 1e-6f);
            Assert.Equal([2.5f, 15.25f], one[4..]);
            Assert.Equal(one, many);
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
            NativeMemory.Free(p);
        }
    }

    /// <summary>
    /// 2³⁰ 個を超える ulong・double の Philox の一括生成は、1個ずつの繰り返しと値・状態が同じ
    /// （語の数が int を超えて OverflowException になっていた）。ブロックの境界と語3から始める
    /// </summary>
    [Theory]
    [InlineData(4)]
    [InlineData(3)]
    public void PhiloxBulkFillBeyondIntWordCountMatchesOneAtATime(int wordIndex)
    {
        if (!Enabled)
        {
            return;
        }

        int n = (1 << 30) + 3;
        var start = new Philox4x32State(1, 2, 7, wordIndex);
        ulong* p = (ulong*)NativeMemory.Alloc((nuint)n, sizeof(ulong));
        try
        {
            var bulk = new Philox4x32Random(start);
            var single = new Philox4x32Random(start);
            var values = new Span<ulong>(p, n);
            bulk.Fill(values);
            for (int i = 0; i < n; i++)
            {
                if (values[i] != single.NextUInt64())
                {
                    Assert.Fail($"ulong の {i} 番目が違う");
                }
            }

            Assert.Equal(single.GetState(), bulk.GetState());

            var doubles = new Span<double>(p, n);
            bulk.Fill(doubles);
            for (int i = 0; i < n; i++)
            {
                if (doubles[i] != single.NextDouble())
                {
                    Assert.Fail($"double の {i} 番目が違う");
                }
            }

            Assert.Equal(single.GetState(), bulk.GetState());
        }
        finally
        {
            NativeMemory.Free(p);
        }
    }

    /// <summary>
    /// 長さが int の上限に近い uint の Philox の一括生成は、1個ずつの繰り返しと値・状態が同じで、領域の外に書かない
    /// （AVX2 で4ブロックずつ求める経路の終了条件 <c>g + 16 ≤ 長さ</c> が溢れ、範囲外に書いていた）
    /// </summary>
    [Fact]
    public void PhiloxBulkFillOfLongestUInt32SpanMatchesOneAtATime()
    {
        if (!Enabled)
        {
            return;
        }

        int n = int.MaxValue - 15; // 16 語（4ブロック）の倍数で、最後の4ブロックを書いた後の g + 16 が溢れる
        var start = new Philox4x32State(1, 2, 7, 4);
        uint* p = (uint*)NativeMemory.Alloc((nuint)n + 16, sizeof(uint));
        try
        {
            var guard = new Span<uint>(p + n, 16);
            guard.Fill(0xDEADBEEF);
            var bulk = new Philox4x32Random(start);
            var single = new Philox4x32Random(start);
            var values = new Span<uint>(p, n);
            bulk.Fill(values);
            Assert.Equal(-1, guard.IndexOfAnyExcept(0xDEADBEEF));
            for (int i = 0; i < n; i++)
            {
                if (values[i] != single.NextUInt32())
                {
                    Assert.Fail($"uint の {i} 番目が違う");
                }
            }

            Assert.Equal(single.GetState(), bulk.GetState());
        }
        finally
        {
            NativeMemory.Free(p);
        }
    }

    /// <summary>
    /// 長さが int の上限に近い float の <see cref="Blas.Swap(Span{float}, Span{float})"/> が、交換を終えた後で例外にならない
    /// （4 KB ずつの区切りの位置 <c>i += 1024</c> が最後の区切りの後に負へ折り返し、長さ 2,147,482,625 以上で ArgumentOutOfRangeException になっていた）。
    /// x と y に同じ領域を渡す（自身との交換は値を変えない）
    /// </summary>
    [Theory]
    [InlineData(2147482624)]
    [InlineData(2147482625)]
    [InlineData(int.MaxValue)]
    public void SwapOfLongFloatVectorReachesTheEnd(int length)
    {
        if (!Enabled)
        {
            return;
        }

        float* p = (float*)AllocateRepeated((nuint)length * sizeof(float), out nuint mapped);
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = 1;
            var x = new Span<float>(p, length);
            x[length - 1] = 3;
            Blas.Swap(x, x);
            Assert.Equal(3, x[length - 1]);
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
            Release(p, mapped);
        }
    }

    /// <summary><see cref="SwapOfLongFloatVectorReachesTheEnd"/> の double 版（区切りは 512 要素で、長さ 2,147,483,137 以上で折り返していた）</summary>
    [Theory]
    [InlineData(2147483136)]
    [InlineData(2147483137)]
    [InlineData(int.MaxValue)]
    public void SwapOfLongDoubleVectorReachesTheEnd(int length)
    {
        if (!Enabled)
        {
            return;
        }

        double* p = (double*)AllocateRepeated((nuint)length * sizeof(double), out nuint mapped);
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = 1;
            var x = new Span<double>(p, length);
            x[length - 1] = 3;
            Blas.Swap(x, x);
            Assert.Equal(3, x[length - 1]);
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
            Release(p, mapped);
        }
    }

    // 長さが int の上限に近い 0 ベクトルの LogSumExp が ln n になることを検証する（末尾の 1024 要素区切りで走査位置が int 上限を超えないこと）。
    // float は約 8 GiB、double は約 16 GiB の仮想領域を使用する。
    [Theory]
    [InlineData(2147482624, 1)]
    [InlineData(2147482625, 1)]
    [InlineData(int.MaxValue, 1)]
    [InlineData(2147482624, 4)]
    [InlineData(2147482625, 4)]
    [InlineData(int.MaxValue, 4)]
    public void LogSumExpOfLongZeroFloatVectorIsLogOfLength(int length, int threads) => CheckLogSumExpOfZeros<float>(length, threads);

    /// <summary><see cref="LogSumExpOfLongZeroFloatVectorIsLogOfLength"/> の double 版</summary>
    [Theory]
    [InlineData(2147482624, 1)]
    [InlineData(2147482625, 1)]
    [InlineData(int.MaxValue, 1)]
    [InlineData(2147482624, 4)]
    [InlineData(2147482625, 4)]
    [InlineData(int.MaxValue, 4)]
    public void LogSumExpOfLongZeroDoubleVectorIsLogOfLength(int length, int threads) => CheckLogSumExpOfZeros<double>(length, threads);

    private static void CheckLogSumExpOfZeros<T>(int length, int threads) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (!Enabled)
        {
            return;
        }

        T* p = (T*)AllocateRepeated((nuint)length * (nuint)sizeof(T), out nuint mapped);
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = threads;
            T actual = XMath.LogSumExp(new ReadOnlyVectorSpan<T>(new ReadOnlySpan<T>(p, length)));
            // 解析的な値は ln n。式の丸めの 1 ulp の差を許す
            T expected = T.Log(T.CreateChecked(length));
            Assert.True(T.IsFinite(actual));
            Assert.InRange(actual, T.BitDecrement(expected), T.BitIncrement(expected));
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
            Release(p, mapped);
        }
    }

    /// <summary>
    /// 0 で埋めた書き込み可能な <paramref name="bytes"/> バイトの領域。書き込むと物理メモリを使うので、Linux では 256 MiB の memfd を
    /// 仮想領域に繰り返し写像する（256 MiB ごとに同じ内容が見える）。ほかでは普通に確保する（<paramref name="mapped"/> は 0）
    /// </summary>
    private static void* AllocateRepeated(nuint bytes, out nuint mapped)
    {
        if (!OperatingSystem.IsLinux())
        {
            mapped = 0;
            return NativeMemory.AllocZeroed(bytes);
        }

        const nuint Period = 256 << 20;
        mapped = (bytes + Period - 1) / Period * Period;

        // PROT_NONE、MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE で仮想領域だけを取り、その上を MAP_SHARED | MAP_FIXED で埋める
        nint address = Mmap(0, mapped, 0, 0x4022, -1, 0);
        if (address == -1)
        {
            throw new InvalidOperationException($"mmap failed: {Marshal.GetLastPInvokeError()}");
        }

        int fd;
        fixed (byte* name = "tuna-large-memory\0"u8)
        {
            fd = MemfdCreate(name, 0);
        }

        try
        {
            if (fd < 0)
            {
                throw new InvalidOperationException($"memfd_create failed: {Marshal.GetLastPInvokeError()}");
            }

            if (Ftruncate(fd, (long)Period) != 0)
            {
                throw new InvalidOperationException($"ftruncate failed: {Marshal.GetLastPInvokeError()}");
            }

            for (nuint offset = 0; offset < mapped; offset += Period)
            {
                // PROT_READ | PROT_WRITE、MAP_SHARED | MAP_FIXED
                if (Mmap(address + (nint)offset, Period, 3, 0x11, fd, 0) == -1)
                {
                    throw new InvalidOperationException($"mmap failed: {Marshal.GetLastPInvokeError()}");
                }
            }
        }
        catch
        {
            Munmap(address, mapped);
            throw;
        }
        finally
        {
            if (fd >= 0)
            {
                Close(fd);
            }
        }

        return (void*)address;
    }

    private static void Release(void* p, nuint mapped)
    {
        if (mapped == 0)
        {
            NativeMemory.Free(p);
        }
        else
        {
            Munmap((nint)p, mapped);
        }
    }

    [DllImport("libc", EntryPoint = "mmap", SetLastError = true)]
    private static extern nint Mmap(nint address, nuint length, int protection, int flags, int fd, nint offset);

    [DllImport("libc", EntryPoint = "munmap", SetLastError = true)]
    private static extern int Munmap(nint address, nuint length);

    [DllImport("libc", EntryPoint = "memfd_create", SetLastError = true)]
    private static extern int MemfdCreate(byte* name, uint flags);

    [DllImport("libc", EntryPoint = "ftruncate", SetLastError = true)]
    private static extern int Ftruncate(int fd, long length);

    [DllImport("libc", EntryPoint = "close", SetLastError = true)]
    private static extern int Close(int fd);
}
