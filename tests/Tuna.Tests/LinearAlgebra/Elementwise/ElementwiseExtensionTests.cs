using Tuna.LinearAlgebra;
using static Tuna.XMath;
using DoubleVector = Tuna.LinearAlgebra.Vector<double>;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>要素演算の拡張: 2・3入力の Zip、ブロードキャストビュー、スカラーとの演算、Softmax・SoftThreshold、ReLU 系の活性化関数</summary>
[Trait("LinearAlgebra", "XMath")]
public class ElementwiseExtensionTests
{
    private static Matrix<double> Sequence(int rows, int columns, double start = 1) =>
        Matrix<double>.FromColumnMajor(rows, columns, Enumerable.Range(0, rows * columns).Select(i => start + i).ToArray());

    private static void AssertClose(double expected, double actual, double relative = 1e-15) =>
        Assert.True(Math.Abs(actual - expected) <= relative * Math.Abs(expected), $"expected {expected:R}, actual {actual:R}");

    private static void AssertClose(IEnumerable<double> expected, ReadOnlyVectorSpan<double> actual, double absolute)
    {
        var e = expected.ToArray();
        Assert.Equal(e.Length, actual.Length);
        for (int i = 0; i < e.Length; i++)
        {
            Assert.True(Math.Abs(actual[i] - e[i]) <= absolute, $"[{i}] expected {e[i]:R}, actual {actual[i]:R}");
        }
    }

    private readonly struct Atan2 : IBinaryOperator<double>
    {
        public static double Invoke(double y, double x) => Math.Atan2(y, x);
    }

    private readonly struct Gcd : IBinaryOperator<int>
    {
        public static int Invoke(int x, int y)
        {
            while (y != 0)
            {
                (x, y) = (y, x % y);
            }

            return x;
        }
    }

    private readonly struct Weighted(double weight) : IStatefulBinaryOperator<double>
    {
        public double Invoke(double x, double y) => weight * x + (1 - weight) * y;
    }

    private readonly struct Lerp : ITernaryOperator<double>
    {
        public static double Invoke(double x, double y, double t) => x + (y - x) * t;
    }

    private readonly struct ScaledFma(double scale) : IStatefulTernaryOperator<double>
    {
        public double Invoke(double x, double y, double z) => scale * (x * y + z);
    }

    // ---------- Zip ----------

    [Fact]
    public void ZipAppliesBinaryOperatorsElementwise()
    {
        var y = new DoubleVector([1, -1, 0.5]);
        var x = new DoubleVector([2, 3, -4]);
        var expected = y.ToArray().Zip(x.ToArray(), Math.Atan2).ToArray();
        Assert.Equal(expected, Zip<Atan2>(y, x).ToArray());
        Assert.Equal(expected, Zip<double, Atan2>(y, x).ToArray());
        var destination = new DoubleVector(3);
        Zip<Atan2>(y, x, destination);                                   // 3つ目の引数が出力先（3入力で結果を新規確保するオーバーロードではない）
        Assert.Equal(expected, destination.ToArray());

        Assert.Equal(new[] { 4, 1, 6 }, Zip<int, Gcd>(new[] { 12, 7, 18 }.ToVector(), new[] { 8, 5, 24 }.ToVector()).ToArray());

        var weighted = y.ToArray().Zip(x.ToArray(), (a, b) => 0.25 * a + 0.75 * b).ToArray();
        Assert.Equal(weighted, Zip(y, x, new Weighted(0.25)).ToArray());
        Assert.Same(destination, Zip(y, x, new Weighted(0.25), destination));
        Assert.Equal(weighted, destination.ToArray());

        // 行列とストライドのあるビュー
        var a = Sequence(2, 3, -2);
        var b = Sequence(2, 3, 5);
        var expectedMatrix = Zip<Atan2>(a, b);
        Assert.Equal(expectedMatrix.ToArray(), Zip<Atan2>(a.T, b.T).T.ToArray());
        Assert.Equal(expectedMatrix.Row(1).ToArray(), Zip<Atan2>(a.Row(1), b.Row(1)).ToArray());
        var c = new Matrix<double>(2, 3);
        Assert.Same(c, Zip(a, b, new Weighted(0.5), c));
        Assert.Equal(new double[,] { { 1.5, 3.5, 5.5 }, { 2.5, 4.5, 6.5 } }, c.ToArray());
    }

    [Fact]
    public void ZipAppliesTernaryOperatorsElementwise()
    {
        var x = new DoubleVector([1, -1, 0.5]);
        var y = new DoubleVector([2, 3, -4]);
        var t = new DoubleVector([0, 0.5, 1]);
        Assert.Equal(new[] { 1.0, 1, -4 }, Zip<Lerp>(x, y, t).ToArray());
        Assert.Equal(new[] { 1.0, 1, -4 }, Zip<double, Lerp>(x, y, t).ToArray());
        var destination = new DoubleVector(3);
        Zip<Lerp>(x, y, t, destination);
        Assert.Equal(new[] { 1.0, 1, -4 }, destination.ToArray());

        var fma = new[] { 2 * (1 * 2 + 0.0), 2 * (-1 * 3 + 0.5), 2 * (0.5 * -4 + 1) };
        Assert.Equal(fma, Zip(x, y, t, new ScaledFma(2)).ToArray());
        Assert.Same(destination, Zip(x, y, t, new ScaledFma(2), destination));
        Assert.Equal(fma, destination.ToArray());

        var a = Sequence(3, 2);
        var b = Sequence(3, 2, 10);
        var w = Matrix<double>.FromColumnMajor(3, 2, [0, 0.5, 1, 1, 0.5, 0]);
        Assert.Equal(new double[,] { { 1, 13 }, { 6.5, 9.5 }, { 12, 6 } }, Zip<Lerp>(a, b, w).ToArray());
        Assert.Equal(Zip<Lerp>(a, b, w).ToArray(), Zip<Lerp>(a.T, b.T, w.T).T.ToArray());
    }

    [Fact]
    public void ZipAllowsExactAliasButRejectsPartialOverlapAndShapeMismatch()
    {
        var x = new DoubleVector([1, 2, 3, 4]);
        var t = new DoubleVector([0.5, 0.5, 0.5, 0.5]);
        Zip<Lerp>(x, x, t, x);
        Assert.Equal(new double[] { 1, 2, 3, 4 }, x.ToArray());
        Zip(x, t, new Weighted(0), x);                                   // x ← t（出力先が第1入力と同じ）
        Assert.Equal(t.ToArray(), x.ToArray());

        var v = new DoubleVector([1, 2, 3, 4]);
        Assert.Throws<ArgumentException>(() => { Zip<Atan2>(v[..3], v[..3], v[1..]); });
        Assert.Throws<ArgumentException>(() => { Zip<Lerp>(v[1..], v[..3], v[..3], v[..3]); });
        Assert.Throws<ArgumentException>(() => { Zip<Atan2>(v, v[..3]); });
        Assert.Throws<ArgumentException>(() => { Zip<Lerp>(v, v, v[..3]); });
        Assert.Throws<ArgumentException>(() => { Zip<Lerp>(Sequence(2, 2), Sequence(2, 2), Sequence(2, 2), new Matrix<double>(2, 3)); });
    }

    // ---------- ブロードキャストビューとスカラーとの演算 ----------

    [Fact]
    public void BroadcastViewsRepeatAVectorAcrossColumnsOrRows()
    {
        var v = new DoubleVector([1, 2]);
        Assert.Equal(new double[,] { { 1, 1, 1 }, { 2, 2, 2 } }, v.Broadcast(3).ToArray());
        Assert.Equal(new double[,] { { 1, 2 }, { 1, 2 }, { 1, 2 } }, v.BroadcastRows(3).ToArray());
        var a = Sequence(2, 3);                                          // 行 0 は [1, 3, 5]
        Assert.Equal(new double[,] { { 1, 1 }, { 3, 3 }, { 5, 5 } }, a.Row(0).Broadcast(2).ToArray());
        Assert.Equal(new double[,] { { 1, 3, 5 } }, a.Row(0).BroadcastRows(1).ToArray());
        Assert.Equal(0, v.Broadcast(0).Columns);
        Assert.Throws<ArgumentOutOfRangeException>(() => { _ = v.Broadcast(-1); });
        Assert.Throws<ArgumentOutOfRangeException>(() => { _ = v.AsVectorSpan().BroadcastRows(-1); });

        // 各行から行の平均を引く（インプレース）、各列を列ごとの係数で拡大する
        var x = Sequence(3, 4);
        var mean = RowSums(x);
        InPlace.Multiply(mean, 0.25);
        var centered = Subtract(x, mean.Broadcast(x.Columns));
        Assert.Equal(new double[] { 0, 0, 0 }, RowSums(centered).ToArray());
        Subtract(x, mean.Broadcast(x.Columns), x);
        Assert.Equal(centered.ToArray(), x.ToArray());
        var scale = new DoubleVector([1, 10, 100, 1000]);
        Assert.Equal(new double[,] { { 1, 40, 700, 10000 }, { 2, 50, 800, 11000 }, { 3, 60, 900, 12000 } },
            Hadamard(Sequence(3, 4), scale.BroadcastRows(3)).ToArray());

        // 出力先と同じ要素をブロードキャストして読むのは部分的な重なり
        var b = Sequence(3, 3);
        Assert.Throws<ArgumentException>(() => { Subtract(b, b.Column(0).Broadcast(3), b); });
    }

    [Fact]
    public void ScalarAddAndSubtractHaveAllocatingDestinationAndInPlaceVersions()
    {
        var x = new DoubleVector([1, 2, 3]);
        Assert.Equal(new[] { 1.5, 2.5, 3.5 }, Add(x, 0.5).ToArray());
        Assert.Equal(new[] { 0.5, 1.5, 2.5 }, Subtract(x, 0.5).ToArray());
        var destination = new DoubleVector(3);
        Add(x, 1.0, destination);
        Assert.Equal(new double[] { 2, 3, 4 }, destination.ToArray());
        Subtract(x, 1.0, destination);
        Assert.Equal(new double[] { 0, 1, 2 }, destination.ToArray());

        var a = Sequence(2, 2);
        Assert.Equal(new double[,] { { 0, 2 }, { 1, 3 } }, Subtract(a, 1.0).ToArray());
        Assert.Equal(new double[,] { { 11, 13 }, { 12, 14 } }, Add(a, 10.0).ToArray());
        var c = new Matrix<double>(2, 2);
        Add(a.T, 1.0, c);
        Assert.Equal(new double[,] { { 2, 3 }, { 4, 5 } }, c.ToArray());
        Subtract(a, 1.0, a);
        Assert.Equal(new double[,] { { 0, 2 }, { 1, 3 } }, a.ToArray());
        InPlace.Subtract(a, 1.0);
        InPlace.Subtract(x, 1.0);
        Assert.Equal(new double[,] { { -1, 1 }, { 0, 2 } }, a.ToArray());
        Assert.Equal(new double[] { 0, 1, 2 }, x.ToArray());

        Assert.Equal(new[] { 2, 3 }, Add(new[] { 1, 2 }.ToVector(), 1).ToArray());
        Assert.Throws<ArgumentException>(() => { Add(x[..2], 1.0, x[1..]); });
    }

    // ---------- Softmax ----------

    [Fact]
    public void SoftmaxNormalizesAndDoesNotOverflow()
    {
        var x = new DoubleVector([1, 2, 3]);
        double e0 = Math.Exp(-2), e1 = Math.Exp(-1), sum = 0 + e0 + e1 + 1;
        Assert.Equal(new[] { e0 / sum, e1 / sum, 1 / sum }, Softmax(x).ToArray());
        Assert.Equal(1.0, Sum(Softmax(x)), 15);

        // 最大値を引くので、大きな値・小さな値でも溢れずに同じ結果になる
        Assert.Equal(Softmax(x).ToArray(), Softmax(Add(x, 1000.0)).ToArray());
        Assert.Equal(new[] { 0.5, 0.5 }, Softmax(new DoubleVector([-1000, -1000])).ToArray());
        Assert.Equal(new[] { 0.0, 1.0 }, Softmax(new DoubleVector([-1e300, 1e300])).ToArray());

        Assert.All(Softmax(new DoubleVector([1, double.NaN, 2])).ToArray(), v => Assert.True(double.IsNaN(v)));
        Assert.Equal(0, Softmax(new DoubleVector(0)).Length);

        var f = Softmax(new float[] { 0, 1, 2, 3 }.ToVector());
        Assert.Equal(1f, Sum(f), 6);

        var destination = new DoubleVector(3);
        Assert.Same(destination, Softmax(x, destination));
        Assert.Equal(Softmax(x).ToArray(), destination.ToArray());
        var expected = Softmax(x).ToArray();
        InPlace.Softmax(x);
        Assert.Equal(expected, x.ToArray());

        var v = new DoubleVector([1, 2, 3, 4]);
        Assert.Throws<ArgumentException>(() => { Softmax(v[..3], v[1..]); });
        Assert.Throws<ArgumentException>(() => { Softmax(v, new DoubleVector(3)); });
    }

    [Fact]
    public void ColumnAndRowSoftmaxAgreeWithTheVectorVersion()
    {
        var a = Matrix<double>.FromColumnMajor(3, 4, [0.5, -1, 2, 3, 3, 3, -7, 0, 7, 1e3, 1e3 + 1, -1e3]);
        var columns = ColumnSoftmax(a);
        var rows = RowSoftmax(a);
        for (int j = 0; j < a.Columns; j++)
        {
            Assert.Equal(Softmax(a.Column(j)).ToArray(), columns.Column(j).ToArray());
        }

        for (int i = 0; i < a.Rows; i++)
        {
            Assert.Equal(Softmax(a.Row(i)).ToArray(), rows.Row(i).ToArray());
        }

        Assert.Equal(rows.ToArray(), ColumnSoftmax(a.T).T.ToArray());
        Assert.Equal(columns.ToArray(), RowSoftmax(a.T).T.ToArray());

        var destination = new Matrix<double>(3, 4);
        Assert.Same(destination, ColumnSoftmax(a, destination));
        Assert.Same(destination, RowSoftmax(a, destination));
        Assert.Equal(rows.ToArray(), destination.ToArray());
        var b = a.Clone();
        InPlace.ColumnSoftmax(b);
        Assert.Equal(columns.ToArray(), b.ToArray());
        InPlace.RowSoftmax(a);
        Assert.Equal(rows.ToArray(), a.ToArray());

        Assert.Equal(0, ColumnSoftmax(new Matrix<double>(0, 3)).Rows);
        Assert.Equal(0, RowSoftmax(new Matrix<double>(3, 0)).Columns);
        Assert.Throws<ArgumentException>(() => { RowSoftmax(b, new Matrix<double>(4, 3)); });
        Assert.Throws<ArgumentException>(() => { ColumnSoftmax(b[.., ..3], b[.., 1..]); });
    }

    // ---------- SoftThreshold ----------

    [Fact]
    public void SoftThresholdShrinksTowardZero()
    {
        var x = new DoubleVector([-3, -1, -0.5, 0, 0.5, 1, 3, double.NaN, double.PositiveInfinity, double.NegativeInfinity]);
        Assert.Equal(new[] { -2, 0, 0, 0, 0, 0, 2, double.NaN, double.PositiveInfinity, double.NegativeInfinity }, SoftThreshold(x, 1.0).ToArray());
        Assert.Equal(x.ToArray(), SoftThreshold(x, 0.0).ToArray());
        Assert.Equal(x.ToArray(), SoftThreshold(x, -0.0).ToArray());
        Assert.Equal(new double[7], SoftThreshold(x[..7], double.PositiveInfinity).ToArray());

        // 整数でも桁あふれしない
        Assert.Equal(new[] { -3, 0, 0, 0, 3 }, SoftThreshold(new[] { -5, -2, 0, 2, 5 }.ToVector(), 2).ToArray());
        Assert.Equal(new[] { int.MinValue + 1, int.MaxValue - 1 }, SoftThreshold(new[] { int.MinValue, int.MaxValue }.ToVector(), 1).ToArray());
        Assert.Equal(new[] { -1, 0 }, SoftThreshold(new[] { int.MinValue, int.MaxValue }.ToVector(), int.MaxValue).ToArray());
        Assert.Equal(new uint[] { 0, 0, 3 }, SoftThreshold(new uint[] { 0, 2, 5 }.ToVector(), 2u).ToArray());

        var a = Matrix<double>.FromColumnMajor(2, 2, [-2, 0.5, 4, -0.25]);
        Assert.Equal(new double[,] { { -1.5, 3.5 }, { 0, 0 } }, SoftThreshold(a, 0.5).ToArray());
        var destination = new Matrix<double>(2, 2);
        SoftThreshold(a.T, 0.5, destination);
        Assert.Equal(new double[,] { { -1.5, 0 }, { 3.5, 0 } }, destination.ToArray());
        InPlace.SoftThreshold(a, 0.5);
        Assert.Equal(new double[,] { { -1.5, 3.5 }, { 0, 0 } }, a.ToArray());
        var y = new DoubleVector([2, -2]);
        SoftThreshold(y, 1.0, y);
        InPlace.SoftThreshold(y, 0.5);
        Assert.Equal(new[] { 0.5, -0.5 }, y.ToArray());

        Assert.Throws<ArgumentOutOfRangeException>(() => { SoftThreshold(x, -1.0); });
        Assert.Throws<ArgumentOutOfRangeException>(() => { SoftThreshold(x, double.NaN); });
        Assert.Throws<ArgumentOutOfRangeException>(() => { InPlace.SoftThreshold(a, -1e-300); });
    }

    // 定義どおりに書いた参照（実装の分岐とは独立）。NaN はビットごとそのまま返す
    private static double SoftReference(double x, double t) => double.IsNaN(x) ? x : x > t ? x - t : x < -t ? x + t : 0.0;

    private static float SoftReference(float x, float t) => float.IsNaN(x) ? x : x > t ? x - t : x < -t ? x + t : 0f;

    private static T SoftReference<T>(T x, T t) where T : System.Numerics.IBinaryInteger<T>
    {
        // Int128 なら桁あふれしない
        Int128 a = Int128.CreateChecked(x), b = Int128.CreateChecked(t);
        return T.CreateChecked(a > b ? a - b : a < -b ? a + b : Int128.Zero);
    }

    private static readonly double NegativeNaN = BitConverter.Int64BitsToDouble(unchecked((long)0xFFF8_0000_0000_1234));
    private static readonly double SignalingNaN = BitConverter.Int64BitsToDouble(0x7FF0_0000_0000_0001);

    private static double[] SoftThresholdPool(double t)
    {
        var random = new Random(31);
        var pool = new List<double>
        {
            0.0, -0.0, 1, -1, 0.5, -0.5, 3, -3, double.Epsilon, -double.Epsilon, double.MaxValue, -double.MaxValue,
            double.PositiveInfinity, double.NegativeInfinity, double.NaN, NegativeNaN, SignalingNaN,
        };
        foreach (double s in new[] { t, -t })
        {
            pool.Add(s);
            pool.Add(Math.BitIncrement(s));
            pool.Add(Math.BitDecrement(s));
        }

        for (int i = 0; i < 16; i++)
        {
            pool.Add(8 * random.NextDouble() - 4);
        }

        return pool.ToArray();
    }

    [Fact]
    public void SoftThresholdMatchesTheDefinitionBitwiseAcrossSimdWidths()
    {
        // 長さを 0〜70 と変えて SIMD の本体と端の両方を通し、特別な値がどちらにも入るよう並びをずらす。
        // ±0 の符号・NaN の符号と内容まで一致させる
        foreach (double t in new[] { 0.0, -0.0, 0.75, 1, double.Epsilon, 1e300, double.MaxValue, double.PositiveInfinity })
        {
            var pool = SoftThresholdPool(t);
            for (int n = 0; n <= 70; n++)
            {
                var x = Enumerable.Range(0, n).Select(i => pool[(7 * i + n) % pool.Length]).ToArray();
                var expected = x.Select(v => BitConverter.DoubleToInt64Bits(SoftReference(v, t))).ToArray();

                Assert.Equal(expected, SoftThreshold(new DoubleVector(x), t).ToArray().Select(BitConverter.DoubleToInt64Bits));
                var inPlace = new DoubleVector(x);
                InPlace.SoftThreshold(inPlace, t);
                Assert.Equal(expected, inPlace.ToArray().Select(BitConverter.DoubleToInt64Bits));
                if (n % 5 == 0)
                {
                    var matrix = Matrix<double>.FromColumnMajor(5, n / 5, x);
                    Assert.Equal(expected, SoftThreshold(matrix, t).AsSpan().ToArray().Select(BitConverter.DoubleToInt64Bits));
                }
            }
        }

        foreach (float t in new[] { 0f, -0f, 0.75f, 1, float.Epsilon, float.MaxValue, float.PositiveInfinity })
        {
            var pool = SoftThresholdPool(t).Select(v => (float)v)
                .Concat([MathF.BitIncrement(t), MathF.BitDecrement(t), MathF.BitIncrement(-t), MathF.BitDecrement(-t), -float.NaN]).ToArray();
            for (int n = 0; n <= 70; n++)
            {
                var x = Enumerable.Range(0, n).Select(i => pool[(7 * i + n) % pool.Length]).ToArray();
                var expected = x.Select(v => BitConverter.SingleToInt32Bits(SoftReference(v, t))).ToArray();
                Assert.Equal(expected, SoftThreshold(x.ToVector(), t).ToArray().Select(BitConverter.SingleToInt32Bits));
            }
        }

        // SIMD のない型は要素ごとの経路を通る
        Assert.Equal(new[] { -2.5m, 0, 0, 0, 1.5m }, SoftThreshold(new[] { -3.5m, -0.5m, 0, 1m, 2.5m }.ToVector(), 1m).ToArray());
        Assert.Equal(new[] { (Half)(-2), Half.Zero, (Half)2, Half.NaN },
            SoftThreshold(new[] { (Half)(-3), (Half)0.5, (Half)3, Half.NaN }.ToVector(), (Half)1).ToArray());
    }

    [Fact]
    public void SoftThresholdOfIntegersDoesNotOverflowAcrossSimdWidths()
    {
        AssertIntegerSoftThreshold<sbyte>();
        AssertIntegerSoftThreshold<byte>();
        AssertIntegerSoftThreshold<short>();
        AssertIntegerSoftThreshold<ushort>();
        AssertIntegerSoftThreshold<int>();
        AssertIntegerSoftThreshold<uint>();
        AssertIntegerSoftThreshold<long>();
        AssertIntegerSoftThreshold<ulong>();
        AssertIntegerSoftThreshold<nint>();
    }

    private static void AssertIntegerSoftThreshold<T>() where T : unmanaged, System.Numerics.IBinaryInteger<T>, System.Numerics.IMinMaxValue<T>
    {
        var random = new Random(37);
        // 1バイトの型は 512 ビットに 64 個入るので、長さは 150 まで見る
        foreach (T t in new[] { T.Zero, T.One, T.CreateTruncating(5), T.MaxValue / T.CreateTruncating(2), T.MaxValue - T.One, T.MaxValue })
        {
            var pool = new List<T> { T.MinValue, T.MinValue + T.One, T.MaxValue, T.MaxValue - T.One, T.Zero, T.One, t, t + T.One, t - T.One };
            // 符号なしでは負の値が大きな値に回り込むが、入力としてはそれでよい
            pool.AddRange(new long[] { -1, -2, -5, -6, -4, 100, -100 }.Select(T.CreateTruncating));
            pool.AddRange(new[] { T.Zero - t, T.Zero - t - T.One, T.Zero - t + T.One });
            pool.AddRange(Enumerable.Range(0, 16).Select(_ => T.CreateTruncating(random.NextInt64(long.MinValue, long.MaxValue))));
            for (int n = 0; n <= 150; n += n < 70 ? 1 : 7)
            {
                var x = Enumerable.Range(0, n).Select(i => pool[(7 * i + n) % pool.Count]).ToArray();
                var expected = x.Select(v => SoftReference(v, t)).ToArray();
                Assert.Equal(expected, SoftThreshold(x.ToVector(), t).ToArray());
            }
        }
    }

    [Fact]
    public void SoftThresholdOnStridedViewsMatchesTheDefinitionAndLeavesTheRestAlone()
    {
        var random = new Random(41);
        double[] values = Enumerable.Range(0, 13 * 11).Select(_ => 6 * random.NextDouble() - 3).ToArray();
        values[20] = double.NaN;
        values[40] = -0.0;
        var original = Matrix<double>.FromColumnMajor(13, 11, values);
        const double t = 1.0;

        void AssertBlock(Matrix<double> actual, int rowStart, int rowEnd, int columnStart, int columnEnd)
        {
            for (int j = 0; j < 11; j++)
            {
                for (int i = 0; i < 13; i++)
                {
                    bool inside = i >= rowStart && i < rowEnd && j >= columnStart && j < columnEnd;
                    double expected = inside ? SoftReference(original[i, j], t) : original[i, j];
                    Assert.Equal(BitConverter.DoubleToInt64Bits(expected), BitConverter.DoubleToInt64Bits(actual[i, j]));
                }
            }
        }

        // 列が連続した部分行列: 新しい行列、別の行列の部分への出力、その場更新
        var sub = SoftThreshold(original[1..12, 2..9], t);
        for (int j = 0; j < 7; j++)
        {
            for (int i = 0; i < 11; i++)
            {
                Assert.Equal(SoftReference(original[i + 1, j + 2], t), sub[i, j]);
            }
        }

        var target = original.Clone();
        SoftThreshold(original[1..12, 2..9], t, target[1..12, 2..9]);
        AssertBlock(target, 1, 12, 2, 9);

        var a = original.Clone();
        InPlace.SoftThreshold(a[1..12, 2..9], t);
        AssertBlock(a, 1, 12, 2, 9);

        // 行が連続した転置ビュー（全体と部分）のその場更新
        a = original.Clone();
        InPlace.SoftThreshold(a.T, t);
        AssertBlock(a, 0, 13, 0, 11);
        a = original.Clone();
        InPlace.SoftThreshold(a.T[2..9, 1..12], t);
        AssertBlock(a, 1, 12, 2, 9);

        // 1行だけの部分行列・間隔のあるベクトル（要素ごとの経路）
        a = original.Clone();
        InPlace.SoftThreshold(a[3..4, 1..10], t);
        AssertBlock(a, 3, 4, 1, 10);
        var strided = new double[2 * 37];
        for (int i = 0; i < strided.Length; i++)
        {
            strided[i] = i % 2 == 0 ? values[i / 2] : 42;
        }

        InPlace.SoftThreshold(new VectorSpan<double>(strided, [(nint)37], [(nint)2]), t);
        for (int i = 0; i < strided.Length; i++)
        {
            Assert.Equal(i % 2 == 0 ? SoftReference(values[i / 2], t) : 42, strided[i]);
        }
    }

    // ---------- ReLU 系 ----------

    private static readonly double[] Points = [-50, -10, -3, -1, -1e-10, -0.0, 0, 1e-10, 0.5, 1, 2, 3, 40];

    [Fact]
    public void ReluFamilyLiftsTheScalarDefinitions()
    {
        var x = Points.ToVector();
        Assert.Equal(Points.Select(v => v > 0 ? v : 0).ToArray(), Relu(x).ToArray());
        Assert.Equal(Points.Select(v => v > 0 ? v : 0.1 * v).ToArray(), LeakyRelu(x, 0.1).ToArray());
        Assert.Equal(Points.Select(v => Relu(v)).ToArray(), Relu(x).ToArray());
        Assert.Equal(Points.Select(v => Elu(v, 1.5)).ToArray(), Elu(x, 1.5).ToArray());
        Assert.Equal(Points.Select(v => Selu(v)).ToArray(), Selu(x).ToArray());
        Assert.Equal(Points.Select(v => Gelu(v)).ToArray(), Gelu(x).ToArray());
        Assert.Equal(Points.Select(v => Gelu(v, true)).ToArray(), Gelu(x, true).ToArray());
        Assert.Equal(Points.Select(v => Silu(v)).ToArray(), Silu(x).ToArray());
        Assert.Equal(Points.Select(v => Softplus(v)).ToArray(), Softplus(x).ToArray());

        // 行列、転置ビュー、destination を受け取るオーバーロード、インプレース版、演算子
        var a = Matrix<double>.FromColumnMajor(2, 3, [-2, -0.5, 0, 0.5, 1, 4]);
        Assert.Equal(new double[,] { { 0, 0, 1 }, { 0, 0.5, 4 } }, Relu(a).ToArray());
        Assert.Equal(Gelu(a).T.ToArray(), Gelu(a.T).ToArray());
        var destination = new Matrix<double>(2, 3);
        Assert.Same(destination, Softplus(a, destination));
        Assert.Equal(Softplus(a).ToArray(), destination.ToArray());
        Assert.Equal(Silu(x).ToArray(), Map<XMath.Operators.Silu<double>>(x).ToArray());
        var geluInPlace = x.ToArray().ToVector();
        InPlace.Gelu(geluInPlace, true);
        Assert.Equal(Gelu(x, true).ToArray(), geluInPlace.ToArray());
        Assert.Equal(Gelu(a, true).T.ToArray(), Gelu(a.T, true).ToArray());
        Assert.Equal(Gelu(a).ToArray(), Gelu(a, false).ToArray());
        Assert.Equal(LeakyRelu(x, 0.2).ToArray(), Map(x, new XMath.Operators.LeakyRelu<double>(0.2)).ToArray());
        var expected = Elu(a, 2.0).ToArray();
        InPlace.Elu(a, 2.0);
        Assert.Equal(expected, a.ToArray());
        InPlace.Relu(x);
        Assert.Equal(Points.Select(v => v > 0 ? v : 0).ToArray(), x.ToArray());

        // 整数・単精度
        Assert.Equal(new[] { 0, 0, 3 }, Relu(new[] { -2, 0, 3 }.ToVector()).ToArray());
        Assert.Equal(new[] { -4, 0, 3 }, LeakyRelu(new[] { -2, 0, 3 }.ToVector(), 2).ToArray());
        Assert.Equal((float)0.84134474606854294859, Gelu(new[] { 1f }.ToVector())[0]);
        // float の tanh 近似は近似版のロジスティック関数で計算する（誤差は数 ulp）
        Assert.Equal((float)0.84119199060827670478, Gelu(new[] { 1f }.ToVector(), true)[0], 6e-7f);
        Assert.Equal((float)1.928749847963917783e-22, Softplus(new[] { -50f }.ToVector())[0]);
    }

    [Fact]
    public void ReluFamilyValuesMatchHighPrecisionReferences()
    {
        // 参照値は mpmath（50桁）で計算した
        AssertClose(0.84134474606854294859, Gelu(1.0));
        AssertClose(-0.15865525393145705141, Gelu(-1.0));
        AssertClose(-7.619853024160526066e-23, Gelu(-10.0), 1e-14);
        AssertClose(0.34573123063700655182, Gelu(0.5));
        AssertClose(2.9959503059051097164, Gelu(3.0));
        // tanh 近似の GELU（approximate: true）は数式 (x/2)·(1 + tanh(√(2/π)·(x + 0.044715·x³))) の値（定数は丸めない）。
        // 負の側は指数の引数 z の丸めが |z| 倍に広がる（x = -10 で z ≈ -87）
        AssertClose(0.84119199060827670478, Gelu(1.0, true));
        AssertClose(-0.15880800939172329522, Gelu(-1.0, true));
        AssertClose(0.34571400982514392204, Gelu(0.5, true));
        AssertClose(2.9963626079182269812, Gelu(3.0, true));
        AssertClose(-0.0036373920817730188378, Gelu(-3.0, true), 1e-14);
        AssertClose(-1.2040923482098059937e-37, Gelu(-10.0, true), 1e-13);
        AssertClose(5.0000000003989422804e-11, Gelu(1e-10, true));
        Assert.True(Math.Abs(Gelu(2.7, true) - Gelu(2.7)) < 4.8e-4);
        AssertClose(1.7615941559557648881, Silu(2.0));
        AssertClose(-0.26894142136999512075, Silu(-1.0));
        AssertClose(-9.6437492398195889151e-21, Silu(-50.0));
        AssertClose(0.69314718055994530942, Softplus(0.0));
        AssertClose(1.928749847963917783e-22, Softplus(-50.0));
        AssertClose(2.1269280110429724964, Softplus(2.0));
        Assert.Equal(40.0, Softplus(40.0));
        AssertClose(1.0507009873554804934, Selu(1.0));
        AssertClose(-1.1113307378125627618, Selu(-1.0));
        AssertClose(-1.7580993407594718929e-10, Selu(-1e-10));
        AssertClose(-9.999999999500000000016667e-11, Elu(-1e-10, 1.0));   // eˣ - 1 をそのまま計算すると約 8 桁しか合わない

        // オーバーフロー・無限大・NaN
        Assert.Equal(1000.0, Softplus(1000.0));
        Assert.Equal(0.0, Softplus(-1000.0));
        Assert.Equal(double.PositiveInfinity, Softplus(double.PositiveInfinity));
        Assert.Equal(-0.0, Gelu(double.NegativeInfinity));
        Assert.Equal(double.PositiveInfinity, Gelu(double.PositiveInfinity));
        Assert.Equal(-0.0, Silu(double.NegativeInfinity));
        Assert.Equal(-0.0, Gelu(double.NegativeInfinity, true));
        Assert.Equal(double.PositiveInfinity, Gelu(double.PositiveInfinity, true));
        Assert.Equal(-0.0, Gelu(-1e200, true));
        Assert.Equal(1e200, Gelu(1e200, true));
        Assert.True(double.IsNegative(Gelu(-0.0, true)));
        Assert.Equal(-1.5, Elu(double.NegativeInfinity, 1.5));
        Assert.Equal(double.NegativeInfinity, LeakyRelu(double.NegativeInfinity, 0.01));
        Assert.True(double.IsPositive(Relu(-0.0)));
        foreach (double v in new[] { Relu(double.NaN), LeakyRelu(double.NaN, 0.1), Elu(double.NaN, 1.0), Selu(double.NaN), Gelu(double.NaN), Gelu(double.NaN, true), Silu(double.NaN), Softplus(double.NaN) })
        {
            Assert.True(double.IsNaN(v));
        }
    }
    // スカラー・連続配列・対角ビューを同じ独立参照値と比較する。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditIdentities20261102")]
    [Theory]
    [InlineData(0, -30.0, -1.472014178144456e-196)]
    [InlineData(0, -38.0, -1.096462777e-314)]
    [InlineData(0, -38.25, -7.9548e-319)]
    [InlineData(0, -38.4, -2.535e-321)]
    [InlineData(0, -38.5, -5.4e-323)]
    [InlineData(0, -38.55, -1e-323)]
    [InlineData(1, -30.0, -1.472014178144456e-196)]
    [InlineData(1, -38.0, -1.096462777e-314)]
    [InlineData(1, -38.25, -7.9548e-319)]
    [InlineData(1, -38.4, -2.535e-321)]
    [InlineData(1, -38.5, -5.4e-323)]
    [InlineData(1, -38.55, -1e-323)]
    [InlineData(2, -30.0, -1.472014178144456e-196)]
    [InlineData(2, -38.0, -1.096462777e-314)]
    [InlineData(2, -38.25, -7.9548e-319)]
    [InlineData(2, -38.4, -2.535e-321)]
    [InlineData(2, -38.5, -5.4e-323)]
    [InlineData(2, -38.55, -1e-323)]
    public void GeluPreservesRepresentableNegativeTails(int layout, double x, double expected)
    {
        void Check(double actual) => SpecialFunctionAssert.Close(expected, actual,
            relativeTolerance: 2e-12, absoluteTolerance: 2 * double.Epsilon);

        if (layout == 0)
        {
            Check(XMath.Gelu(x));
            return;
        }

        // SIMD幅を越える長さと、間隔のある入力の両方を通す。
        const int length = 17;
        if (layout == 1)
        {
            var input = new DoubleVector(Enumerable.Repeat(x, length).ToArray());
            var actual = XMath.Gelu(input);
            for (int i = 0; i < length; i++)
            {
                Check(actual[i]);
            }
        }
        else
        {
            var input = new Matrix<double>(length, length);
            for (int i = 0; i < length; i++)
            {
                input[i, i] = x;
            }

            var actual = XMath.Gelu(input.Diagonal());
            for (int i = 0; i < length; i++)
            {
                Check(actual[i]);
            }
        }
    }

    // 真の値が -0 へ丸まる範囲では、x² がオーバーフローする大きさでも NaN にせず -0 を返す（-39 は丸めの境界 -38.6 付近より外側）
    [Theory]
    [InlineData(0, -39.0)]
    [InlineData(0, -1e154)]
    [InlineData(0, -1e155)]
    [InlineData(0, -1e200)]
    [InlineData(0, -double.MaxValue)]
    [InlineData(1, -1e155)]
    [InlineData(1, -1e200)]
    [InlineData(1, -double.MaxValue)]
    [InlineData(2, -1e155)]
    [InlineData(2, -1e200)]
    [InlineData(2, -double.MaxValue)]
    public void GeluUnderflowsToNegativeZeroBeforeSquaring(int layout, double x)
    {
        static void Check(double actual)
        {
            Assert.Equal(0.0, actual);
            Assert.True(double.IsNegative(actual));
        }

        const int length = 17;
        if (layout == 0)
        {
            Check(XMath.Gelu(x));
        }
        else if (layout == 1)
        {
            var actual = XMath.Gelu(new DoubleVector(Enumerable.Repeat(x, length).ToArray()));
            for (int i = 0; i < length; i++)
            {
                Check(actual[i]);
            }
        }
        else
        {
            var input = new Matrix<double>(length, length);
            for (int i = 0; i < length; i++)
            {
                input[i, i] = x;
            }

            var actual = XMath.Gelu(input.Diagonal());
            for (int i = 0; i < length; i++)
            {
                Check(actual[i]);
            }
        }
    }

    // スカラー・連続配列・対角ビューを同じ独立参照値と比較する。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditIdentities20261102")]
    [Theory]
    [InlineData(0, -700.0, -6.90177358063184e-302)]
    [InlineData(0, -710.0, -3.1781632202293424e-306)]
    [InlineData(0, -720.0, -1.46320617774547e-310)]
    [InlineData(0, -740.0, -3.09967e-319)]
    [InlineData(0, -750.0, -1.5e-323)]
    [InlineData(1, -700.0, -6.90177358063184e-302)]
    [InlineData(1, -710.0, -3.1781632202293424e-306)]
    [InlineData(1, -720.0, -1.46320617774547e-310)]
    [InlineData(1, -740.0, -3.09967e-319)]
    [InlineData(1, -750.0, -1.5e-323)]
    [InlineData(2, -700.0, -6.90177358063184e-302)]
    [InlineData(2, -710.0, -3.1781632202293424e-306)]
    [InlineData(2, -720.0, -1.46320617774547e-310)]
    [InlineData(2, -740.0, -3.09967e-319)]
    [InlineData(2, -750.0, -1.5e-323)]
    public void SiluPreservesRepresentableNegativeTails(int layout, double x, double expected)
    {
        void Check(double actual) => SpecialFunctionAssert.Close(expected, actual,
            relativeTolerance: 2e-12, absoluteTolerance: 2 * double.Epsilon);

        if (layout == 0)
        {
            Check(XMath.Silu(x));
            return;
        }

        // SIMD幅を越える長さと、間隔のある入力の両方を通す。
        const int length = 17;
        if (layout == 1)
        {
            var input = new DoubleVector(Enumerable.Repeat(x, length).ToArray());
            var actual = XMath.Silu(input);
            for (int i = 0; i < length; i++)
            {
                Check(actual[i]);
            }
        }
        else
        {
            var input = new Matrix<double>(length, length);
            for (int i = 0; i < length; i++)
            {
                input[i, i] = x;
            }

            var actual = XMath.Silu(input.Diagonal());
            for (int i = 0; i < length; i++)
            {
                Check(actual[i]);
            }
        }
    }
}
