using System.Numerics;
using System.Runtime.CompilerServices;

namespace Tuna.Tests.LinearAlgebra;

using Tuna.LinearAlgebra;

[Collection(nameof(ParallelismCollection))]
public class BroadcastElementwiseTests
{
    [Fact]
    public void BroadcastRowsMatchesScalarBitsAcrossLayoutsAndDegrees()
    {
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            foreach (int degree in new[] { 1, 2, 3, 4 })
            {
                LinearAlgebraSettings.MaxDegreeOfParallelism = degree;
                CheckLayouts<double>();
                CheckLayouts<float>();
            }
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    private static void CheckLayouts<T>() where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
    {
        T[] pool = [T.Zero, T.NegativeZero, T.One, -T.One, T.Epsilon, -T.Epsilon,
            T.MaxValue, T.MinValue, T.PositiveInfinity, T.NegativeInfinity, T.NaN,
            FromBits<T>(0x7ff8000000000042, 0x7fc00042), FromBits<T>(0xfff8000000001234, 0xffc01234),
            FromBits<T>(0x7ff0000000000001, 0x7f800001), T.CreateChecked(1.25)];
        var random = new Random(103);
        // 端数が SIMD 幅をまたぐ形と、空・長さ1を含む。
        foreach (int rows in new[] { 0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 33, 65 })
        {
            foreach (int columns in new[] { 0, 1, 19 })
            {
                foreach (int layout in Enumerable.Range(0, 6))
                {
                    foreach (bool scalarOnLeft in new[] { false, true })
                    {
                        foreach (int operation in Enumerable.Range(0, 6))
                        {
                            int rowStride = layout == 3 ? 2 : layout == 2 ? columns + 3 : 1;
                            int columnStride = layout == 2 ? 1 : rows * rowStride + 3;
                            var input = new T[2 * (rows + 3) * (columns + 3)];
                            var output = new T[input.Length];
                            var sentinel = T.CreateChecked(123);
                            Array.Fill(output, sentinel);
                            var x = new MatrixSpan<T>(input.AsSpan(1), [rows, columns], [rowStride, columnStride]);
                            // layout 4 は入力と出力の向きが異なる一般経路。
                            int outputRowStride = layout == 4 ? columns + 3 : rowStride;
                            int outputColumnStride = layout == 4 ? 1 : columnStride;
                            var d = new MatrixSpan<T>(output.AsSpan(1), [rows, columns], [outputRowStride, outputColumnStride]);

                            var scalars = new T[2 * columns + 1];
                            var v = new VectorSpan<T>(scalars.AsSpan(1), [columns], [2]);
                            for (int j = 0; j < columns; j++)
                            {
                                v[j] = pool[j % pool.Length];
                                for (int i = 0; i < rows; i++)
                                {
                                    x[i, j] = i < pool.Length ? pool[(i + j) % pool.Length] : T.CreateChecked(random.NextDouble() * 8 - 4);
                                }
                            }

                            var expected = new T[rows * columns];
                            for (int j = 0; j < columns; j++)
                            {
                                for (int i = 0; i < rows; i++)
                                {
                                    expected[i + j * rows] = scalarOnLeft ? Scalar(operation, v[j], x[i, j]) : Scalar(operation, x[i, j], v[j]);
                                }
                            }

                            if (layout == 5)
                            {
                                d = x;
                            }

                            var broadcast = v.AsReadOnly().BroadcastRows(rows);
                            // 転置した場合は、行ごとのスカラー演算の経路。
                            bool transpose = layout == 2;
                            var left = scalarOnLeft ? broadcast : x.AsReadOnly();
                            var right = scalarOnLeft ? x.AsReadOnly() : broadcast;
                            Into(operation, transpose ? left.T : left, transpose ? right.T : right, transpose ? d.T : d);
                            for (int j = 0; j < columns; j++)
                            {
                                for (int i = 0; i < rows; i++)
                                {
                                    // 入力が2つとも NaN の場合のペイロードと、sNaN の quiet 化は既存の SIMD 契約と同じ扱い。
                                    bool manyNaNs = T.IsNaN(v[j]) && (i < pool.Length && T.IsNaN(pool[(i + j) % pool.Length]));
                                    AssertBits(expected[i + j * rows], d[i, j], manyNaNs,
                                        $"{typeof(T).Name} rows={rows}, columns={columns}, layout={layout}, left={scalarOnLeft}, op={operation}, ({i},{j})");
                                }
                            }

                            // 出力ビューの外側の要素は書き換えない。
                            if (layout != 5)
                            {
                                var touched = new bool[output.Length];
                                for (int j = 0; j < columns; j++)
                                {
                                    for (int i = 0; i < rows; i++)
                                    {
                                        touched[1 + i * (int)d.Strides[0] + j * (int)d.Strides[1]] = true;
                                    }
                                }

                                for (int k = 0; k < output.Length; k++)
                                {
                                    if (!touched[k])
                                    {
                                        Assert.Equal(sentinel, output[k]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private static T FromBits<T>(ulong doubleBits, uint floatBits) where T : unmanaged =>
        typeof(T) == typeof(double) ? Unsafe.BitCast<ulong, T>(doubleBits) : Unsafe.BitCast<uint, T>(floatBits);

    private static void AssertBits<T>(T expected, T actual, bool manyNaNs, string context)
        where T : unmanaged, INumberBase<T>
    {
        if (manyNaNs)
        {
            Assert.True(T.IsNaN(expected) && T.IsNaN(actual), context);
            return;
        }

        ulong Bits(T value) => typeof(T) == typeof(double) ? Unsafe.BitCast<T, ulong>(value) : Unsafe.BitCast<T, uint>(value);
        ulong quiet = typeof(T) == typeof(double) ? 1UL << 51 : 1UL << 22;
        ulong e = Bits(expected), a = Bits(actual);
        if (T.IsNaN(expected))
        {
            e |= quiet;
        }

        if (T.IsNaN(actual))
        {
            a |= quiet;
        }

        Assert.True(e == a, $"{context}: {e:X16} != {a:X16}");
    }

    private static T Scalar<T>(int operation, T x, T y) where T : INumber<T> => operation switch
    {
        0 => x + y,
        1 => x - y,
        2 => x * y,
        3 => x / y,
        4 => T.Min(x, y),
        5 => T.Max(x, y),
        _ => throw new ArgumentOutOfRangeException(nameof(operation)),
    };

    private static void Into<T>(int operation, ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, MatrixSpan<T> d)
        where T : unmanaged, INumber<T>
    {
        switch (operation)
        {
            case 0: XMath.Add(x, y, d); break;
            case 1: XMath.Subtract(x, y, d); break;
            case 2: XMath.Hadamard(x, y, d); break;
            case 3: XMath.Divide(x, y, d); break;
            case 4: XMath.Min(x, y, d); break;
            case 5: XMath.Max(x, y, d); break;
        }
    }

    [Fact]
    public void BroadcastDoesNotBypassShapeAndOverlapChecks()
    {
        var a = new Matrix<double>(17, 8);
        var v = new Vector<double>(8);
        Assert.Throws<ArgumentException>(() => XMath.Subtract(a, v.BroadcastRows(16), a));
        Assert.Throws<ArgumentException>(() => XMath.Subtract(a, v.BroadcastRows(17), new Matrix<double>(17, 7)));
        Assert.Throws<ArgumentException>(() => XMath.Subtract(a, a.Row(0).BroadcastRows(17), a));
        Assert.Throws<ArgumentException>(() => XMath.Subtract(a.Row(0).BroadcastRows(17), a, a));
        Assert.Throws<ArgumentException>(() => XMath.Subtract(a[..16, ..], v.BroadcastRows(16), a[1.., ..]));
    }

    private readonly struct DifferenceWithOffset(double offset) : IStatefulBinaryOperator<double>
    {
        public double Invoke(double x, double y) => x - y + offset;
    }

    [Fact]
    public void UnsupportedOperatorsAndTypesKeepTheGeneralPath()
    {
        var a = new Matrix<double>(17, 3);
        var v = new Vector<double>([1, 2, 3]);
        var result = XMath.Zip(a, v.BroadcastRows(17), new DifferenceWithOffset(0.5));
        Assert.Equal(-2.5, result[16, 2]);
        var integers = new Matrix<int>(17, 3);
        var iv = new Vector<int>([1, 2, 3]);
        Assert.Equal(-3, XMath.Subtract(integers, iv.BroadcastRows(17))[16, 2]);
        var half = new Matrix<Half>(17, 3);
        var hv = new Vector<Half>([(Half)1, (Half)2, (Half)3]);
        Assert.Equal((Half)(-3), XMath.Subtract(half, hv.BroadcastRows(17))[16, 2]);
    }

    [Fact]
    public void DestinationVersionsDoNotAllocateAfterWarmup()
    {
        CheckAllocation<double>();
        CheckAllocation<float>();
    }

    private static void CheckAllocation<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        var a = new Matrix<T>(257, 1000);
        var v = new Vector<T>(1000);
        var d = new Matrix<T>(257, 1000);
        foreach (int operation in Enumerable.Range(0, 6))
        {
            foreach (bool left in new[] { false, true })
            {
                for (int k = 0; k < 32; k++)
                {
                    Call();
                }

                GC.Collect(0, GCCollectionMode.Forced, blocking: true);
                long before = GC.GetAllocatedBytesForCurrentThread();
                for (int k = 0; k < 32; k++)
                {
                    Call();
                }

                Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
                void Call()
                {
                    var b = v.BroadcastRows(257);
                    Into(operation, left ? b : a.AsMatrixSpan().AsReadOnly(), left ? a.AsMatrixSpan().AsReadOnly() : b, d);
                }
            }
        }
    }
}
