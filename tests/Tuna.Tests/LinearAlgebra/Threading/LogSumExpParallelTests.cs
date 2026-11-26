using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using Tuna.LinearAlgebra;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>長い指数和も、区間の端数・最大値の個数・並列数によらず逐次版と同じ値になる</summary>
[Trait("LinearAlgebra", "Parallelism")]
[Collection(nameof(ParallelismCollection))]
public class LogSumExpParallelTests
{
    [Theory]
    [InlineData(524287)]
    [InlineData(524288)]
    [InlineData(524289)]
    [InlineData(525311)]
    [InlineData(525312)]
    [InlineData(525313)]
    [InlineData(589824)]
    [InlineData(1048575)]
    public void BlockChunkAndParallelBoundariesKeepTheSameBits(int length)
    {
        CheckBoundaries<double>(length);
        CheckBoundaries<float>(length);
    }

    private static T[] Values<T>(int length) where T : IFloatingPointIeee754<T>
    {
        var values = new T[length];
        var random = new Random(314159 + length);
        for (int i = 0; i < length; i++)
        {
            values[i] = T.CreateChecked(-80 * random.NextDouble());
        }

        // 同じ最大値を区間の境界と最後の端数にも置く。
        foreach (int index in new[] { 0, 65535, 65536, 524287, length - 1 })
        {
            if (index < length)
            {
                values[index] = T.Zero;
            }
        }

        return values;
    }

    private static void CheckBoundaries<T>(int length) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var input = Values<T>(length);
        var padded = new T[length + 2];
        padded[0] = padded[^1] = T.NaN;
        input.CopyTo(padded, 1);
        WithDegrees(() => XMath.LogSumExp(new ReadOnlyVectorSpan<T>(padded.AsSpan(1, length))));
    }

    /// <summary>最大値の個数と、非常に小さい残りの寄与を、それぞれ独立の解析的な値でも検証する</summary>
    [Fact]
    public void LongSumsKeepTiedMaximaAndTinyContributions()
    {
        const int length = 524289;
        var x = new double[length];
        Array.Fill(x, -40);
        x[^1] = 0;
        WithDegrees(() => XMath.LogSumExp(new ReadOnlyVectorSpan<double>(x)), value =>
        {
            double expected = XMath.Log1p((length - 1) * Math.Exp(-40));
            Assert.InRange(Math.Abs((value - expected) / expected), 0, 5e-15);
        });

        // すべて最大値でも、非最大の指数和と個数を混ぜずに ln n を返す。
        Array.Clear(x);
        WithDegrees(() => XMath.LogSumExp(new ReadOnlyVectorSpan<double>(x)), value =>
            Assert.InRange(value, Math.BitDecrement(Math.Log(length)), Math.BitIncrement(Math.Log(length))));
    }

    [Fact]
    public void LongSpecialValuesPreserveTheContract()
    {
        var x = new double[524289];
        Array.Fill(x, double.NegativeInfinity);
        WithDegrees(() => XMath.LogSumExp(new ReadOnlyVectorSpan<double>(x)), value => Assert.Equal(double.NegativeInfinity, value));
        x[65536] = double.PositiveInfinity;
        WithDegrees(() => XMath.LogSumExp(new ReadOnlyVectorSpan<double>(x)), value => Assert.Equal(double.PositiveInfinity, value));
        x[^1] = double.NaN;
        WithDegrees(() => XMath.LogSumExp(new ReadOnlyVectorSpan<double>(x)), value => Assert.True(double.IsNaN(value)));
    }

    [Fact]
    public void ColumnAndLogSoftmaxPathsKeepTheSameBits()
    {
        CheckRelatedPaths<double>();
        CheckRelatedPaths<float>();
    }

    private static void CheckRelatedPaths<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        const int length = 524353;
        var x = Values<T>(length);
        var matrix = Matrix<T>.FromColumnMajor(length, 1, x);
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = 1;
            T expected = XMath.LogSumExp(matrix.Column(0));
            var expectedSoftmax = XMath.LogSoftmax<T>(matrix.Column(0));
            foreach (int degree in new[] { 2, 3, 4 })
            {
                LinearAlgebraSettings.MaxDegreeOfParallelism = degree;
                AssertBits(expected, XMath.ColumnLogSumExp<T>(matrix)[0]);
                var actual = XMath.ColumnLogSoftmax<T>(matrix);
                for (int i = 0; i < length; i++)
                {
                    AssertBits(expectedSoftmax[i], actual[i, 0]);
                }
            }
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    [Fact]
    public void WidenedHalfStillRoundsTheSingleResultOnce()
    {
        const int length = 524289;
        var x = new Half[length];
        var single = new float[length];
        Half expected = (Half)XMath.LogSumExp(new ReadOnlyVectorSpan<float>(single));
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            foreach (int degree in new[] { 1, 2, 4 })
            {
                LinearAlgebraSettings.MaxDegreeOfParallelism = degree;
                Assert.Equal(expected, XMath.LogSumExp(new ReadOnlyVectorSpan<Half>(x)));
            }
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    [Fact]
    public async Task ConcurrentAndNestedCallsKeepTheSameBits()
    {
        var x = Values<double>(524353);
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = 1;
            double expected = XMath.LogSumExp(new ReadOnlyVectorSpan<double>(x));
            LinearAlgebraSettings.MaxDegreeOfParallelism = 4;
            await Task.WhenAll(Enumerable.Range(0, 4).Select(_ => Task.Run(() =>
            {
                for (int call = 0; call < 4; call++)
                {
                    AssertBits(expected, XMath.LogSumExp(new ReadOnlyVectorSpan<double>(x)));
                }
            })));

            // 既存の実行器の並列区間から呼び、内側の逐次フォールバックも検査する。
            var run = typeof(LinearAlgebraSettings).Assembly.GetType("Tuna.LinearAlgebra.Parallelism")!
                .GetMethods(BindingFlags.NonPublic | BindingFlags.Static)
                .Single(m => m.Name == "ForStatic" && !m.IsGenericMethod)
                .CreateDelegate<Action<int, Action<int>>>();
            run(4, _ => AssertBits(expected, XMath.LogSumExp(new ReadOnlyVectorSpan<double>(x))));
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    private static void WithDegrees<T>(Func<T> compute, Action<T>? check = null) where T : unmanaged, IFloatingPointIeee754<T>
    {
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = 1;
            T expected = compute();
            check?.Invoke(expected);
            foreach (int degree in new[] { 2, 3, 4 })
            {
                LinearAlgebraSettings.MaxDegreeOfParallelism = degree;
                AssertBits(expected, compute());
            }
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    private static void AssertBits<T>(T expected, T actual) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (T.IsNaN(expected))
        {
            Assert.True(T.IsNaN(actual));
        }
        else if (Unsafe.SizeOf<T>() == sizeof(double))
        {
            Assert.Equal(Unsafe.BitCast<T, long>(expected), Unsafe.BitCast<T, long>(actual));
        }
        else
        {
            Assert.Equal(Unsafe.BitCast<T, int>(expected), Unsafe.BitCast<T, int>(actual));
        }
    }
}
