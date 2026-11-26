using System.Security.Cryptography;
using System.Text.Json;
using Tuna.LinearAlgebra;
using Tuna.QuasiRandom;
using Tuna.RandomSource;

namespace Tuna.Tests.QuasiRandom;

public class SobolSequenceTests
{
    private const double Scale = 4503599627370496.0; // 2^52

    private static JsonDocument Reference() => JsonDocument.Parse(
        typeof(SobolSequenceTests).Assembly.GetManifestResourceStream("Tuna.Tests.QuasiRandom.sobol-reference.json")!);

    // 座標 (k + 1/2)·2^-52 の k
    private static ulong Digits(double u) => (ulong)(u * Scale);

    private static string Sha256(Matrix<double> points)
    {
        var bytes = new byte[points.Rows * points.Columns * sizeof(ulong)];
        int offset = 0;
        for (int i = 0; i < points.Columns; i++)
        {
            for (int j = 0; j < points.Rows; j++, offset += sizeof(ulong))
            {
                BitConverter.TryWriteBytes(bytes.AsSpan(offset), Digits(points[j, i]));
            }
        }

        return Convert.ToHexStringLower(SHA256.HashData(bytes));
    }

    [Fact]
    public void UnscrambledPointsMatchIndependentReference()
    {
        using var document = Reference();
        var root = document.RootElement;
        int dimension = root.GetProperty("Dimension").GetInt32();
        Assert.Equal(SobolSequence.MaxDimension, dimension);
        var sobol = new SobolSequence(dimension);

        var prefix = sobol.GetPoints(0, root.GetProperty("PrefixCount").GetInt32());
        Assert.Equal(root.GetProperty("PrefixSha256").GetString(), Sha256(prefix));

        int index = 0;
        foreach (var expected in root.GetProperty("FirstPoints").EnumerateArray())
        {
            int j = 0;
            foreach (var value in expected.EnumerateArray())
            {
                Assert.Equal(value.GetUInt64(), Digits(prefix[j++, index]));
            }

            index++;
        }

        foreach (var entry in root.GetProperty("Large").EnumerateArray())
        {
            var point = new Matrix<double>(dimension, 1);
            sobol.GetPoints(ulong.Parse(entry.GetProperty("Index").GetString()!), point.AsMatrixSpan());
            Assert.Equal(entry.GetProperty("Sha256").GetString(), Sha256(point));
            int j = 0;
            foreach (var value in entry.GetProperty("Head").EnumerateArray())
            {
                Assert.Equal(value.GetUInt64(), Digits(point[j++, 0]));
            }
        }
    }

    [Fact]
    public void FirstDimensionIsVanDerCorputInGrayCodeOrder()
    {
        var points = new SobolSequence(3).GetPoints(0, 4096);
        for (ulong i = 0; i < 4096; i++)
        {
            ulong gray = i ^ (i >> 1);
            ulong reversed = 0;
            for (int bit = 0; bit < 64; bit++)
            {
                reversed |= ((gray >> bit) & 1) << (63 - bit);
            }

            Assert.Equal(((reversed >> 12) + 0.5) / Scale, points[0, (int)i]);
        }
    }

    [Fact]
    public void ValuesStayStrictlyInsideTheUnitInterval()
    {
        var sobol = new SobolSequence(SobolSequence.MaxDimension);
        var first = new double[sobol.Dimension];
        sobol.GetPoint(0, first);
        Assert.All(first, u => Assert.Equal(Math.ScaleB(1, -53), u));

        var last = new double[sobol.Dimension];
        sobol.GetPoint(ulong.MaxValue, last);
        Assert.All(last, u => Assert.InRange(u, Math.ScaleB(1, -53), 1 - Math.ScaleB(1, -53)));

        var scrambled = new SobolSequence(64, new Philox4x32Random(7)).GetPoints(0, 4096);
        Assert.All(scrambled.AsSpan().ToArray(), u => Assert.True(u > 0 && u < 1));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void EveryCoordinateOfTheFirstPowerOfTwoPointsIsStratified(bool scrambled)
    {
        // 各座標は (0, m, 1)-ネット: 最初の 2^m 点は長さ 2^-m の区間に1点ずつ入る
        var sobol = scrambled ? new SobolSequence(SobolSequence.MaxDimension, new Philox4x32Random(11)) : new SobolSequence(SobolSequence.MaxDimension);
        var points = sobol.GetPoints(0, 1024);
        for (int m = 0; m <= 10; m++)
        {
            int n = 1 << m;
            for (int j = 0; j < sobol.Dimension; j++)
            {
                var seen = new bool[n];
                for (int i = 0; i < n; i++)
                {
                    int cell = (int)(points[j, i] * n);
                    Assert.False(seen[cell], $"dimension {j}, m = {m}");
                    seen[cell] = true;
                }
            }
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FirstTwoDimensionsFormZeroTNets(bool scrambled)
    {
        // 第0・第1次元は (0, m, 2)-ネット: 面積 2^-m のどの基本区間にも1点ずつ入る
        var sobol = scrambled ? new SobolSequence(2, new Philox4x32Random(3)) : new SobolSequence(2);
        var points = sobol.GetPoints(0, 4096);
        for (int m = 1; m <= 12; m++)
        {
            int n = 1 << m;
            for (int a = 0; a <= m; a++)
            {
                var seen = new bool[n];
                for (int i = 0; i < n; i++)
                {
                    int cell = ((int)(points[0, i] * (1 << a)) << (m - a)) | (int)(points[1, i] * (1 << (m - a)));
                    Assert.False(seen[cell], $"m = {m}, a = {a}");
                    seen[cell] = true;
                }
            }
        }
    }

    [Fact]
    public void TwoDimensionalProjectionsOfTheLeadingDimensionsHaveSmallTValues()
    {
        // 方向数の表の品質: 先頭32次元のどの2次元射影も、最初の 2^m 点（m ≤ 12）が (t, m, 2)-ネットで t ≤ 6
        const int dimension = 32, maxM = 12, maxT = 6;
        var points = new SobolSequence(dimension).GetPoints(0, 1 << maxM);
        for (int i = 0; i < dimension; i++)
        {
            for (int j = i + 1; j < dimension; j++)
            {
                for (int m = 1; m <= maxM; m++)
                {
                    int t = TValue(points, i, j, m);
                    Assert.True(t <= maxT, $"dimensions {i} and {j}, m = {m}: t = {t}");
                }
            }
        }
    }

    // 最初の 2^m 点が (t, m, 2)-ネット（a + b = m - t のどの基本区間 2^-a × 2^-b にも 2^t 点ずつ入る）になる最小の t
    private static int TValue(Matrix<double> points, int i, int j, int m)
    {
        int n = 1 << m;
        for (int t = 0; t < m; t++)
        {
            int k = m - t;
            bool net = true;
            var counts = new int[1 << k];
            for (int a = 0; a <= k && net; a++)
            {
                Array.Clear(counts);
                for (int p = 0; p < n; p++)
                {
                    counts[((int)(points[i, p] * (1 << a)) << (k - a)) | (int)(points[j, p] * (1 << (k - a)))]++;
                }

                net = Array.TrueForAll(counts, c => c == 1 << t);
            }

            if (net)
            {
                return t;
            }
        }

        return m;
    }

    [Fact]
    public void ChunkedAndSinglePointGenerationMatchOneCall()
    {
        foreach (var sobol in new[] { new SobolSequence(37), new SobolSequence(300, new Philox4x32Random(5)) })
        {
            var whole = sobol.GetPoints(0, 777);

            var chunked = new Matrix<double>(sobol.Dimension, 777);
            int start = 0;
            foreach (int length in new[] { 1, 2, 5, 120, 0, 256, 393 })
            {
                sobol.GetPoints((ulong)start, chunked[.., start..(start + length)]);
                start += length;
            }

            Assert.Equal(whole.AsSpan().ToArray(), chunked.AsSpan().ToArray());

            var point = new double[sobol.Dimension];
            foreach (int i in new[] { 0, 1, 2, 3, 255, 256, 511, 776 })
            {
                sobol.GetPoint((ulong)i, point);
                for (int j = 0; j < sobol.Dimension; j++)
                {
                    Assert.Equal(whole[j, i], point[j]);
                }
            }

            // ストライドのあるビュー（大きな行列の一部）にも書ける
            var larger = new Matrix<double>(sobol.Dimension + 3, 20);
            sobol.GetPoints(500, larger[1..(sobol.Dimension + 1), 2..12]);
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < sobol.Dimension; j++)
                {
                    Assert.Equal(whole[j, 500 + i], larger[j + 1, i + 2]);
                }
            }
        }
    }

    [Fact]
    public void ScramblingIsReproducibleAndDependsOnTheRandomSource()
    {
        var a = new SobolSequence(16, new Philox4x32Random(42)).GetPoints(0, 256);
        var b = new SobolSequence(16, new Philox4x32Random(42)).GetPoints(0, 256);
        var c = new SobolSequence(16, new Philox4x32Random(43)).GetPoints(0, 256);
        Assert.Equal(a.AsSpan().ToArray(), b.AsSpan().ToArray());
        Assert.NotEqual(a.AsSpan().ToArray(), c.AsSpan().ToArray());
        Assert.True(new SobolSequence(16, new Philox4x32Random(42)).IsScrambled);
        Assert.False(new SobolSequence(16).IsScrambled);

        // 乱数源から次元ごとに64個の64ビット整数を取る
        var random = new Philox4x32Random(42);
        _ = new SobolSequence(16, random);
        var reference = new Philox4x32Random(42);
        for (int i = 0; i < 16 * 64; i++)
        {
            reference.NextUInt64();
        }

        Assert.Equal(reference.NextUInt64(), random.NextUInt64());
    }

    [Fact]
    public void ScrambledEstimatesAreUnbiasedAndFarMoreAccurateThanMonteCarlo()
    {
        // f(x) = Π (1 + (x_j - 1/2)/j)、積分は 1、分散は Π (1 + 1/(12 j²)) - 1
        const int dimension = 16, count = 4096, replicates = 16;
        double variance = 1;
        for (int j = 1; j <= dimension; j++)
        {
            variance *= 1 + 1.0 / (12.0 * j * j);
        }

        variance -= 1;
        double monteCarloError = Math.Sqrt(variance / count);

        double squaredError = 0;
        for (int r = 0; r < replicates; r++)
        {
            var points = new SobolSequence(dimension, new Philox4x32Random(1000, (ulong)r)).GetPoints(0, count);
            double sum = 0;
            for (int i = 0; i < count; i++)
            {
                double f = 1;
                for (int j = 0; j < dimension; j++)
                {
                    f *= 1 + (points[j, i] - 0.5) / (j + 1);
                }

                sum += f;
            }

            squaredError += Math.Pow(sum / count - 1, 2);
        }

        double rootMeanSquareError = Math.Sqrt(squaredError / replicates);
        Assert.True(rootMeanSquareError < monteCarloError / 20, $"RMSE {rootMeanSquareError:E3}, Monte Carlo {monteCarloError:E3}");
    }

    [Fact]
    public void InvalidArgumentsAreRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new SobolSequence(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SobolSequence(SobolSequence.MaxDimension + 1));
        Assert.Throws<ArgumentNullException>(() => new SobolSequence(3, null!));

        var sobol = new SobolSequence(3);
        Assert.Throws<ArgumentException>(() => sobol.GetPoint(0, new double[2]));
        Assert.Throws<ArgumentException>(() => sobol.GetPoints(0, new Matrix<double>(4, 2).AsMatrixSpan()));
        Assert.Throws<ArgumentOutOfRangeException>(() => sobol.GetPoints(0, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => sobol.GetPoints(ulong.MaxValue, 2));
        Assert.Equal(1, sobol.GetPoints(ulong.MaxValue, 1).Columns);
        Assert.Equal(0, sobol.GetPoints(ulong.MaxValue, 0).Columns);
        Assert.Equal(1, sobol.GetPoints(ulong.MaxValue - 1, 2).Columns - 1);
    }
}
