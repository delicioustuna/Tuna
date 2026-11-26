using System.Numerics;
using Tuna.LinearAlgebra;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>ベクトルの集約（内積・ノルム・絶対値の和）は、最大並列数を変えても値のビット表現を保つ</summary>
[Trait("LinearAlgebra", "Parallelism")]
[Collection(nameof(ParallelismCollection))]
public class Level1DeterminismTests
{
    /// <summary>設定する最大並列数（論理プロセッサ数を超える値は、実際の並列数が論理プロセッサ数になる）</summary>
    private static readonly int[] Degrees = [1, 2, 3, 4, 5, 7, 8, 16, 33];

    /// <summary>
    /// 並列化しない長さ、並列化の下限（型と演算によって 16384〜131072）の前後、区間の境界に端数がある長さ、区間の数が上限に達する長さ
    /// </summary>
    public static TheoryData<int> Lengths => new()
    {
        1000, 16_383, 16_384, 16_385, 32_767, 32_768, 32_769, 65_535, 65_536, 65_537,
        100_000, 131_071, 131_072, 131_073, 300_007, 1_000_003, 2_200_001, 4_400_003,
    };

    [Theory]
    [MemberData(nameof(Lengths))]
    public void DoubleReductionsHaveIdenticalBits(int n)
    {
        double[] x = Values(n, 4100 + n);
        double[] y = Values(n, 4200 + n);
        AcrossThreadCounts(() => [Bits(Blas.Dot(x, y)), Bits(Blas.Norm2(x)), Bits(Blas.SumAbs(x))]);
    }

    [Theory]
    [MemberData(nameof(Lengths))]
    public void SingleReductionsHaveIdenticalBits(int n)
    {
        float[] x = Array.ConvertAll(Values(n, 4300 + n), v => (float)v);
        float[] y = Array.ConvertAll(Values(n, 4400 + n), v => (float)v);
        AcrossThreadCounts(() => [Bits(Blas.Dot(x, y)), Bits(Blas.Norm2(x)), Bits(Blas.SumAbs(x))]);
    }

    [Theory]
    [MemberData(nameof(Lengths))]
    public void ComplexReductionsHaveIdenticalBits(int n)
    {
        Complex[] x = ComplexValues(n, 4500 + n);
        Complex[] y = ComplexValues(n, 4600 + n);
        AcrossThreadCounts(() =>
        {
            Complex dot = Blas.Dot(x, y);
            Complex conjugate = Blas.DotConjugate(x, y);
            return [Bits(dot.Real), Bits(dot.Imaginary), Bits(conjugate.Real), Bits(conjugate.Imaginary), Bits(Blas.Norm2(x)), Bits(Blas.SumAbs(x))];
        });
    }

    /// <summary>
    /// 小さい整数の値では、どの順に足しても丸めが起きない。区間に分けても、すべての要素をちょうど1回ずつ数える
    /// </summary>
    [Theory]
    [MemberData(nameof(Lengths))]
    public void ReductionsOfSmallIntegersAreExact(int n)
    {
        var x = new double[n];
        var y = new double[n];
        long dot = 0;
        long squares = 0;
        long abs = 0;
        for (int i = 0; i < n; i++)
        {
            int a = (i % 3) - 1;
            int b = 1 + (i % 2);
            x[i] = a;
            y[i] = b;
            dot += a * b;
            squares += a * a;
            abs += Math.Abs(a);
        }

        float[] xs = Array.ConvertAll(x, v => (float)v);
        float[] ys = Array.ConvertAll(y, v => (float)v);
        Complex[] xc = Array.ConvertAll(x, v => new Complex(v, -v));
        Complex[] yc = Array.ConvertAll(y, v => new Complex(v, v));
        AcrossThreadCounts(() =>
        {
            Assert.Equal((double)dot, Blas.Dot(x, y));
            Assert.Equal(Math.Sqrt(squares), Blas.Norm2(x));
            Assert.Equal((double)abs, Blas.SumAbs(x));
            Assert.Equal((float)dot, Blas.Dot(xs, ys));
            Assert.Equal((float)Math.Sqrt(squares), Blas.Norm2(xs));
            Assert.Equal((float)abs, Blas.SumAbs(xs));

            // (a − ai)(b + bi) = 2ab、共役では (a + ai)(b + bi) = 2ab·i
            Assert.Equal(new Complex(2 * dot, 0), Blas.Dot(xc, yc));
            Assert.Equal(new Complex(0, 2 * dot), Blas.DotConjugate(xc, yc));
            Assert.Equal(Math.Sqrt(2 * squares), Blas.Norm2(xc));
            Assert.Equal((double)(2 * abs), Blas.SumAbs(xc));
            return [];
        });
    }

    /// <summary>区間に分けた和は、要素を順に足した和と丸め誤差の範囲で一致する</summary>
    [Theory]
    [InlineData(100_000)]
    [InlineData(1_000_003)]
    public void ReductionsMatchSequentialSums(int n)
    {
        double[] x = Values(n, 4700 + n);
        double[] y = Values(n, 4800 + n);
        double dot = 0;
        double dotScale = 0;
        double squares = 0;
        double abs = 0;
        for (int i = 0; i < n; i++)
        {
            dot += x[i] * y[i];
            dotScale += Math.Abs(x[i] * y[i]);
            squares += x[i] * x[i];
            abs += Math.Abs(x[i]);
        }

        AcrossThreadCounts(() =>
        {
            Assert.InRange(Math.Abs(Blas.Dot(x, y) - dot), 0, 1e-12 * dotScale);
            Assert.InRange(Math.Abs(Blas.Norm2(x) - Math.Sqrt(squares)), 0, 1e-12 * Math.Sqrt(squares));
            Assert.InRange(Math.Abs(Blas.SumAbs(x) - abs), 0, 1e-12 * abs);
            return [];
        });
    }

    /// <summary>
    /// 長いベクトルのノルムは、平方和がオーバーフロー・アンダーフローする大きさでも正しく、非数と無限大の扱いも並列数によらない
    /// </summary>
    [Theory]
    [InlineData(1e200)]
    [InlineData(1e-200)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(0.0)]
    public void LongNormHandlesExtremeValues(double value)
    {
        const int N = 200_000;
        var all = new double[N];
        Array.Fill(all, value);
        var one = new double[N];
        Array.Fill(one, 1.0);
        one[N - 7] = value;
        double expectedAll = double.IsNaN(value) ? double.NaN : Math.Abs(value) * Math.Sqrt(N);
        double expectedOne = double.IsNaN(value) ? double.NaN
            : double.IsInfinity(value) ? double.PositiveInfinity
            : Math.Abs(value) > 1 ? Math.Abs(value)
            : Math.Sqrt(N - 1);
        AcrossThreadCounts(() =>
        {
            double normAll = Blas.Norm2(all);
            double normOne = Blas.Norm2(one);
            AssertNear(expectedAll, normAll);
            AssertNear(expectedOne, normOne);
            return [Bits(normAll), Bits(normOne)];
        });
    }

    /// <summary>他のスレッドが並列区間を使っていて逐次に実行する呼び出しも、同じビットを返す</summary>
    [Fact]
    public async Task ConcurrentCallersGetIdenticalBits()
    {
        const int N = 300_007;
        double[] x = Values(N, 4900);
        double[] y = Values(N, 4901);
        float[] xs = Array.ConvertAll(x, v => (float)v);
        ulong[] Compute() => [Bits(Blas.Dot(x, y)), Bits(Blas.Norm2(x)), Bits(Blas.SumAbs(x)), Bits(Blas.Norm2(xs))];
        ulong[] expected = Compute();
        var tasks = new Task[4];
        for (int t = 0; t < tasks.Length; t++)
        {
            tasks[t] = Task.Run(() =>
            {
                for (int iteration = 0; iteration < 200; iteration++)
                {
                    Assert.Equal(expected, Compute());
                }
            });
        }

        await Task.WhenAll(tasks).WaitAsync(TimeSpan.FromSeconds(120));
    }

    /// <summary>最大並列数 1 の結果を基準に、並列数を変えた結果のビットが同じことを確かめる</summary>
    private static void AcrossThreadCounts(Func<ulong[]> compute)
    {
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = 1;
            ulong[] expected = compute();
            foreach (int degree in Degrees)
            {
                LinearAlgebraSettings.MaxDegreeOfParallelism = degree;
                ulong[] actual = compute();
                Assert.True(expected.AsSpan().SequenceEqual(actual), $"並列数 {degree}: [{string.Join(", ", actual.Select(v => v.ToString("X16")))}]、並列数 1: [{string.Join(", ", expected.Select(v => v.ToString("X16")))}]");
            }
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    private static void AssertNear(double expected, double actual)
    {
        if (double.IsNaN(expected) || double.IsInfinity(expected) || expected == 0)
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            Assert.InRange(Math.Abs(actual - expected), 0, 1e-13 * expected);
        }
    }

    private static ulong Bits(double value) => BitConverter.DoubleToUInt64Bits(value);

    private static ulong Bits(float value) => BitConverter.SingleToUInt32Bits(value);

    /// <summary>符号と桁がばらつく値（足す順で丸めが変わる）</summary>
    private static double[] Values(int n, int seed)
    {
        var random = new Random(seed);
        var values = new double[n];
        for (int i = 0; i < n; i++)
        {
            values[i] = (random.NextDouble() - 0.5) * Math.Pow(10, random.Next(-3, 4));
        }

        return values;
    }

    private static Complex[] ComplexValues(int n, int seed)
    {
        double[] parts = Values(2 * n, seed);
        var values = new Complex[n];
        for (int i = 0; i < n; i++)
        {
            values[i] = new Complex(parts[2 * i], parts[2 * i + 1]);
        }

        return values;
    }
}
