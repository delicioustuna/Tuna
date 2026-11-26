using Tuna.LinearAlgebra;
using static Tuna.XMath;
using Complex = System.Numerics.Complex;
using DoubleVector = Tuna.LinearAlgebra.Vector<double>;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>縮約（総和の方式・平均・分散・最大最小・内積・ノルム・LogSumExp・LogSoftmax）と要素ごとの初等関数</summary>
[Trait("LinearAlgebra", "XMath")]
public class ReductionAndElementaryTests
{
    private static double[] RandomValues(int count, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, count).Select(_ => (random.NextDouble() - 0.5) * Math.Pow(10, random.Next(-3, 4))).ToArray();
    }

    private static void AssertClose(double expected, double actual, double relative = 1e-15) =>
        Assert.True(Math.Abs(actual - expected) <= relative * Math.Abs(expected), $"expected {expected:R}, actual {actual:R}");

    // NaN どうしも一致とみなし、それ以外は符号つきゼロを含めてビットまで比べる
    private static void AssertSameBits(double expected, double actual)
    {
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual), $"expected NaN, actual {actual:R}");
        }
        else
        {
            Assert.True(BitConverter.DoubleToInt64Bits(expected) == BitConverter.DoubleToInt64Bits(actual), $"expected {expected:R}, actual {actual:R}");
        }
    }

    // 総和の方式（D4）を別の書き方で定義した参照: 64 個ずつのブロックを先頭から順に足し、
    // ブロックの和は「2 の冪の個数の先頭部分（完全二分木）」と「残り（再帰的に同じ規則）」を左右に置いて足す
    private static double ReferenceSum(IReadOnlyList<double> values)
    {
        var blocks = new List<double>();
        for (int start = 0; start < values.Count; start += 64)
        {
            double sum = 0;
            for (int i = start; i < Math.Min(values.Count, start + 64); i++)
            {
                sum += values[i];
            }

            blocks.Add(sum);
        }

        return blocks.Count == 0 ? 0 : Tree(blocks, 0, blocks.Count);
    }

    private static double Tree(List<double> blocks, int start, int count)
    {
        if (count == 1)
        {
            return blocks[start];
        }

        int left = 1 << System.Numerics.BitOperations.Log2((uint)(count - 1));   // count 未満で最大の 2 の冪
        return Tree(blocks, start, left) + Tree(blocks, start + left, count - left);
    }

    // ---------- 総和の方式（D4） ----------

    [Fact]
    public void SumIsSequentialWithinABlockAndATreeAcrossBlocks()
    {
        foreach (int n in new[] { 0, 1, 2, 63, 64, 65, 127, 128, 129, 192, 64 * 5 + 7, 512, 1000, 4097 })
        {
            var values = RandomValues(n, n);
            double sequential = 0;
            foreach (double v in values)
            {
                sequential += v;
            }

            double sum = Sum(values.ToVector());
            if (n <= 64)
            {
                AssertSameBits(sequential, sum);
            }

            AssertSameBits(ReferenceSum(values), sum);
        }

        // 0.1 を 100 万回: 順に足すと 1.3e-6 ずれるが、ブロック単位の二分木では 1e-9 未満に収まる
        var tenths = Enumerable.Repeat(0.1, 1_000_000).ToArray();
        double naive = 0;
        foreach (double v in tenths)
        {
            naive += v;
        }

        Assert.True(Math.Abs(naive - 1e5) > 1e-6);
        Assert.True(Math.Abs(Sum(tenths.ToVector()) - 1e5) < 1e-9);

        Assert.Equal(0.0, Sum(new DoubleVector(0)));
        Assert.Equal(5050, Sum(Enumerable.Range(1, 100).ToArray().ToVector()));
        Assert.Equal(1.5f, Sum(new[] { 0.5f, 1f }.ToVector()));
        Assert.Equal(new Complex(4, 1), Sum(new[] { new Complex(1, 2), new Complex(3, -1) }.ToVector()));
    }

    [Fact]
    public void SumsEqualTheSequentialSumWhenNoAdditionRounds()
    {
        // 部分和がすべて正確に表せる入力（2⁻¹⁰ の倍数で、和の絶対値が 2⁴³ 未満）では、足す順番によらず結果は1つに決まる。
        // 以前の実装（T.Zero から順に足す）とビットまで一致しなければ、和の取り方に誤りがある
        var random = new Random(23);
        foreach (int n in new[] { 65, 1000, 4097, 100_000 })
        {
            var values = Enumerable.Range(0, n).Select(_ => random.Next(-1 << 20, 1 << 20) / 1024.0).ToArray();
            double sequential = 0;
            foreach (double v in values)
            {
                sequential += v;
            }

            AssertSameBits(sequential, Sum(values.ToVector()));

            var a = Matrix<double>.FromColumnMajor(4, n / 4, values[..(n / 4 * 4)]);
            var rows = RowSums(a);
            var columns = ColumnSums(a.T);
            for (int i = 0; i < a.Rows; i++)
            {
                double row = 0;
                for (int j = 0; j < a.Columns; j++)
                {
                    row += a[i, j];
                }

                AssertSameBits(row, rows[i]);
                AssertSameBits(row, columns[i]);
            }
        }

        // 単精度でも同じ（小さな整数なので部分和は 2²⁴ 未満で正確）
        var floats = Enumerable.Range(0, 5000).Select(_ => (float)random.Next(-1000, 1000)).ToArray();
        float sequentialFloat = 0;
        foreach (float v in floats)
        {
            sequentialFloat += v;
        }

        Assert.Equal(sequentialFloat, Sum(floats.ToVector()));
    }

    [Fact]
    public void SumsDoNotDependOnLayoutOrOrientation()
    {
        foreach (int columns in new[] { 0, 1, 64, 65, 256, 333 })
        {
            var a = Matrix<double>.FromColumnMajor(3, columns, RandomValues(3 * columns, columns));
            var rows = RowSums(a);
            var transposed = ColumnSums(a.T);
            for (int i = 0; i < a.Rows; i++)
            {
                var row = a.Row(i);                                      // ストライド 3 のビュー
                AssertSameBits(Sum(row.ToArray().ToVector()), Sum(row));
                AssertSameBits(Sum(row), rows[i]);
                AssertSameBits(rows[i], transposed[i]);
            }

            var sums = ColumnSums(a);
            var transposedRows = RowSums(a.T);
            for (int j = 0; j < columns; j++)
            {
                AssertSameBits(Sum(a.Column(j)), sums[j]);
                AssertSameBits(sums[j], transposedRows[j]);
            }
        }

        // 整数、空の行列、および destination を受け取るオーバーロード
        var m = Matrix<int>.FromColumnMajor(2, 100, Enumerable.Range(0, 200).ToArray());
        Assert.Equal(new[] { 9900, 10000 }, RowSums(m).ToArray());
        Assert.Equal(new double[] { 0, 0, 0 }, ColumnSums(new Matrix<double>(0, 3)).ToArray());
        Assert.Equal(new double[] { 0, 0, 0 }, RowSums(new Matrix<double>(3, 0)).ToArray());
        Assert.Equal(0, RowSums(new Matrix<double>(0, 3)).Length);
        var b = Matrix<double>.FromColumnMajor(2, 3, [1, 2, 3, 4, 5, 6]);
        var destination = new DoubleVector(3);
        Assert.Same(destination, ColumnSums(b, destination));
        Assert.Equal(new double[] { 3, 7, 11 }, destination.ToArray());
        Assert.Throws<ArgumentException>(() => { RowSums(b, destination); });
        Assert.Throws<ArgumentException>(() => { RowSums<double>(b, b.Column(0)); });
    }

    // ---------- 平均・分散 ----------

    [Fact]
    public void MeansAgreeAcrossVectorColumnAndRowForms()
    {
        Assert.Equal(2.5, Mean(new DoubleVector([1, 2, 3, 4])));
        Assert.True(double.IsNaN(Mean(new DoubleVector(0))));
        Assert.Equal(2, Mean(new[] { 1, 2, 4 }.ToVector()));            // 整数型では整数の除算

        var a = Matrix<double>.FromColumnMajor(4, 70, RandomValues(280, 3));
        var rows = RowMeans(a);
        var columns = ColumnMeans(a);
        for (int i = 0; i < a.Rows; i++)
        {
            AssertSameBits(Mean(a.Row(i)), rows[i]);
        }

        for (int j = 0; j < a.Columns; j++)
        {
            AssertSameBits(Mean(a.Column(j)), columns[j]);
        }

        Assert.Equal(rows.ToArray(), ColumnMeans(a.T).ToArray());
        var destination = new DoubleVector(4);
        Assert.Same(destination, RowMeans(a, destination));
        Assert.Throws<ArgumentException>(() => { ColumnMeans(a, destination); });
    }

    [Fact]
    public void VarianceIsAccurateForShiftedDataAndRequiresAnExplicitCorrection()
    {
        // 平均が大きくても桁落ちしない（2乗の平均から平均の2乗を引く方式では大きくずれる）
        var shifted = new DoubleVector([1e9 + 4, 1e9 + 7, 1e9 + 13, 1e9 + 16]);
        Assert.Equal(30.0, Variance(shifted, 1));
        Assert.Equal(22.5, Variance(shifted, 0));
        Assert.InRange(Variance(new DoubleVector([0.1, 0.1, 0.1]), 1), 0.0, 1e-32);   // 丸めで負にならない
        AssertClose(5.0 / 3, Variance(new[] { 1f, 2f, 3f, 4f }.ToVector(), 1), 1e-6);

        // n ≤ correction は NaN、負の補正は例外
        Assert.Equal(0.0, Variance(new DoubleVector([5]), 0));
        Assert.True(double.IsNaN(Variance(new DoubleVector([5]), 1)));
        Assert.True(double.IsNaN(Variance(new DoubleVector(0), 0)));
        Assert.Throws<ArgumentOutOfRangeException>(() => { Variance(shifted, -1); });

        var a = Matrix<double>.FromColumnMajor(4, 70, RandomValues(280, 5).Select(v => v + 1e6).ToArray());
        var rows = RowVariances(a, 1);
        var columns = ColumnVariances(a, 0);
        for (int i = 0; i < a.Rows; i++)
        {
            AssertSameBits(Variance(a.Row(i), 1), rows[i]);
        }

        for (int j = 0; j < a.Columns; j++)
        {
            AssertSameBits(Variance(a.Column(j), 0), columns[j]);
        }

        Assert.Equal(rows.ToArray(), ColumnVariances(a.T, 1).ToArray());
        Assert.All(RowVariances(new Matrix<double>(2, 1), 1).ToArray(), v => Assert.True(double.IsNaN(v)));
        Assert.All(ColumnVariances(new Matrix<double>(1, 2), 1).ToArray(), v => Assert.True(double.IsNaN(v)));
        var destination = new DoubleVector(70);
        Assert.Same(destination, ColumnVariances(a, 1, destination));
        Assert.Throws<ArgumentOutOfRangeException>(() => { RowVariances(a, -1); });
        Assert.Throws<ArgumentOutOfRangeException>(() => { ColumnVariances(a, -1); });
        Assert.Throws<ArgumentException>(() => { RowVariances(a, 1, destination); });
    }

    [Fact]
    public void MeanAndVarianceStayFiniteWhenOnlyIntermediateSumsOverflow()
    {
        // 和は溢れるが平均は表現できる。行・列の形も同じ値になる
        var large = new DoubleVector(Enumerable.Repeat(1e308, 129).ToArray());
        AssertClose(1e308, Mean(large), 1e-15);
        AssertClose(-1e308, Mean(new DoubleVector(Enumerable.Repeat(-1e308, 129).ToArray())), 1e-15);
        Assert.Equal(0.0, Variance(large, 0));
        AssertClose(1e308, Mean(new DoubleVector([1e308, 1.5e308, 0.5e308])), 1e-15);
        AssertClose(1e38f, Mean(Enumerable.Repeat(1e38f, 70).ToArray().ToVector()), 1e-6);

        // 偏差の平方和だけが溢れる（母分散 1e308）。非正規化数の分散も 0 にしない
        AssertClose(1e308, Variance(new DoubleVector([-1e154, 1e154]), 0), 1e-14);
        AssertClose(1e308, Variance(new DoubleVector([-1e154, 1e154, 0]), 1), 1e-14);
        Assert.Equal(1e-320, Variance(new DoubleVector([-1e-160, 1e-160]), 0), 4 * double.Epsilon);

        // 有限の入力でも表現できない平均・分散、∞・NaN を含む入力はそのまま
        Assert.Equal(double.PositiveInfinity, Variance(new DoubleVector([-1e308, 1e308]), 0));
        Assert.Equal(double.PositiveInfinity, Mean(new DoubleVector([1e308, double.PositiveInfinity])));
        Assert.True(double.IsNaN(Mean(new DoubleVector([double.NegativeInfinity, double.PositiveInfinity]))));
        Assert.True(double.IsNaN(Variance(new DoubleVector([1e308, double.NaN]), 0)));

        var a = Matrix<double>.FromColumnMajor(2, 129, Enumerable.Range(0, 258).Select(i => i % 2 == 0 ? 1e308 : (i % 4 == 1 ? -1e154 : 1e154)).ToArray());
        var rowMeans = RowMeans(a);
        var rowVariances = RowVariances(a, 0);
        for (int i = 0; i < a.Rows; i++)
        {
            AssertSameBits(Mean(a.Row(i)), rowMeans[i]);
            AssertSameBits(Variance(a.Row(i), 0), rowVariances[i]);
            Assert.True(double.IsFinite(rowMeans[i]) && double.IsFinite(rowVariances[i]));
        }

        Assert.Equal(rowMeans.ToArray(), ColumnMeans(a.T).ToArray());
        Assert.Equal(rowVariances.ToArray(), ColumnVariances(a.T, 0).ToArray());
    }

    [Fact]
    public void VarianceStaysFiniteWhenOnlyTheSquareOfTheDeviationSumOverflows()
    {
        // 平均が 1e166 程度だと、平均の丸め誤差 δ による Σ(x - 平均) ≈ nδ の2乗だけが溢れる（偏差の平方和は有限）
        var random = new Random(5);
        double c = 1e166;
        var x = Enumerable.Range(0, 100_000).Select(_ => c + (random.NextDouble() - 0.5) * 1e152).ToArray();
        // c との差は丸めなしで求まるので、差の分散を 2 の冪で縮めて参照値にする
        var d = x.Select(t => Math.ScaleB(t - c, -500)).ToArray();
        double dm = d.Average();
        double reference = Math.ScaleB(d.Sum(t => (t - dm) * (t - dm)) / x.Length, 1000);
        double variance = Variance(x.ToVector(), 0);
        AssertClose(reference, variance, 1e-12);
        AssertSameBits(variance, RowVariances(Matrix<double>.FromColumnMajor(1, x.Length, x), 0)[0]);
    }

    // ---------- 最大・最小 ----------

    [Fact]
    public void MaxAndMinPropagateNaNAndRejectEmptyInputs()
    {
        var x = new DoubleVector([3, -1, 7, 2]);
        Assert.Equal(7.0, Max(x));
        Assert.Equal(-1.0, Min(x));
        Assert.Equal(3, Max(new[] { 3, -5, 2 }.ToVector()));
        Assert.Equal(-5, Min(new[] { 3, -5, 2 }.ToVector()));
        Assert.True(double.IsNaN(Max(new DoubleVector([1, double.NaN, 2]))));
        Assert.True(double.IsNaN(Max(new DoubleVector([double.NaN, 1]))));
        Assert.True(double.IsNaN(Min(new DoubleVector([1, 2, double.NaN]))));
        Assert.True(double.IsPositive(Max(new DoubleVector([-0.0, 0.0]))));
        Assert.True(double.IsNegative(Min(new DoubleVector([0.0, -0.0]))));
        Assert.Throws<ArgumentException>(() => { Max(new DoubleVector(0)); });
        Assert.Throws<ArgumentException>(() => { Min(new DoubleVector(0)); });

        var a = Matrix<double>.FromColumnMajor(3, 4, [0.5, -1, 2, 3, 3, 3, -7, double.NaN, 7, 1e3, 1e3 + 1, -1e3]);
        var columnMax = ColumnMax(a);
        var columnMin = ColumnMin(a);
        var rowMax = RowMax(a);
        var rowMin = RowMin(a);
        for (int j = 0; j < a.Columns; j++)
        {
            AssertSameBits(Max(a.Column(j)), columnMax[j]);
            AssertSameBits(Min(a.Column(j)), columnMin[j]);
        }

        for (int i = 0; i < a.Rows; i++)
        {
            AssertSameBits(Max(a.Row(i)), rowMax[i]);
            AssertSameBits(Min(a.Row(i)), rowMin[i]);
        }

        Assert.Equal(RowMax(a).ToArray(), ColumnMax(a.T).ToArray());
        Assert.Equal(new[] { 2, 6 }, RowMax(Matrix<int>.FromColumnMajor(2, 3, [0, 1, 2, 3, -4, 6])).ToArray());
        Assert.Equal(0, ColumnMax(new Matrix<double>(3, 0)).Length);
        Assert.Equal(0, RowMin(new Matrix<double>(0, 3)).Length);
        Assert.Throws<ArgumentException>(() => { ColumnMax(new Matrix<double>(0, 3)); });
        Assert.Throws<ArgumentException>(() => { RowMin(new Matrix<double>(3, 0)); });
        var destination = new DoubleVector(4);
        Assert.Same(destination, ColumnMin(a, destination));
        Assert.Throws<ArgumentException>(() => { RowMax(a, destination); });
    }

    // ---------- 内積・ノルム ----------

    [Fact]
    public void DotSumsProductsWithoutConjugation()
    {
        Assert.Equal(32.0, Dot(new DoubleVector([1, 2, 3]), new DoubleVector([4, 5, 6])));
        Assert.Equal(32, Dot(new[] { 1, 2, 3 }.ToVector(), new[] { 4, 5, 6 }.ToVector()));
        Assert.Equal(new Complex(3, 7), Dot(new[] { new Complex(1, 2), new Complex(3, 0) }.ToVector(), new[] { new Complex(1, -1), new Complex(0, 2) }.ToVector()));
        Assert.Equal(0.0, Dot(new DoubleVector(0), new DoubleVector(0)));
        Assert.Throws<ArgumentException>(() => { Dot(new DoubleVector(2), new DoubleVector(3)); });

        // 長いベクトルとストライドのあるビューでも、積を総和と同じ木で足す
        var a = Matrix<double>.FromColumnMajor(2, 300, RandomValues(600, 11));
        var products = Enumerable.Range(0, 300).Select(j => a[0, j] * a[1, j]).ToArray();
        AssertSameBits(ReferenceSum(products), Dot(a.Row(0), a.Row(1)));
    }

    [Fact]
    public void NormsAvoidOverflowAndUnderflow()
    {
        Assert.Equal(5.0, Norm(new DoubleVector([3, -4])));
        Assert.Equal(5f, Norm(new[] { 3f, 4f }.ToVector()));
        AssertClose(Math.Sqrt(2) * 1e300, Norm(new DoubleVector([1e300, 1e300])));        // 2乗の和は溢れる
        AssertClose(Math.Sqrt(2) * 1e-200, Norm(new DoubleVector([1e-200, -1e-200])));    // 2乗の和は 0 になる
        Assert.Equal(5 * double.Epsilon, Norm(new DoubleVector([3 * double.Epsilon, 4 * double.Epsilon])));
        AssertClose(Math.Sqrt(2) * 1e30f, Norm(new[] { 1e30f, 1e30f }.ToVector()), 3e-7);
        Assert.Equal(double.MaxValue, Norm(new DoubleVector([double.MaxValue, 1])));

        // 特別な値
        Assert.Equal(0.0, Norm(new DoubleVector(0)));
        Assert.Equal(0.0, Norm(new DoubleVector([0, -0.0])));
        Assert.Equal(double.PositiveInfinity, Norm(new DoubleVector([1, double.NegativeInfinity])));
        Assert.True(double.IsNaN(Norm(new DoubleVector([double.PositiveInfinity, double.NaN]))));
        Assert.True(double.IsNaN(Norm(new DoubleVector([double.NaN, double.PositiveInfinity]))));
        Assert.True(double.IsNaN(Norm(new DoubleVector([1, double.NaN]))));

        Assert.Equal(7.0, Norm1(new DoubleVector([3, -4])));
        Assert.Equal(7, Norm1(new[] { -3, 4 }.ToVector()));
        Assert.Equal(new Complex(7, 0), Norm1(new[] { new Complex(3, 4), new Complex(0, -2) }.ToVector()));
        Assert.Equal(7.0, NormInfinity(new DoubleVector([3, -7, 2])));
        Assert.Equal(5, NormInfinity(new[] { -5, 3 }.ToVector()));
        Assert.Equal(0.0, NormInfinity(new DoubleVector(0)));
        Assert.True(double.IsNaN(NormInfinity(new DoubleVector([double.NaN, 7]))));
        Assert.True(double.IsNaN(NormInfinity(new DoubleVector([7, double.NaN]))));

        // 各行・各列のノルムはベクトル版とビットまで一致する（行は複数のブロックにまたがる）
        var values = RandomValues(6 * 70, 13);
        var a = Matrix<double>.FromColumnMajor(6, 70, values);
        for (int j = 0; j < 70; j++)
        {
            a[1, j] *= 1e300;
            a[2, j] *= 1e-300;
            a[5, j] = 0;
        }

        a[2, 3] = double.Epsilon;
        a[3, 10] = double.NaN;
        a[4, 20] = double.NegativeInfinity;
        var rows = RowNorms(a);
        var columns = ColumnNorms(a);
        for (int i = 0; i < a.Rows; i++)
        {
            AssertSameBits(Norm(a.Row(i)), rows[i]);
        }

        for (int j = 0; j < a.Columns; j++)
        {
            AssertSameBits(Norm(a.Column(j)), columns[j]);
        }

        Assert.True(double.IsNaN(rows[3]));
        Assert.Equal(double.PositiveInfinity, rows[4]);
        Assert.Equal(0.0, rows[5]);
        Assert.Equal(new double[] { 0, 0 }, RowNorms(new Matrix<double>(2, 0)).ToArray());
        var destination = new DoubleVector(6);
        Assert.Same(destination, RowNorms(a, destination));
        Assert.Throws<ArgumentException>(() => { ColumnNorms(a, destination); });
    }

    // ---------- 対数領域 ----------

    [Fact]
    public void LogSumExpDoesNotOverflowAndHandlesSpecialValues()
    {
        // 参照値は mpmath（40桁）で計算した
        AssertClose(3.4076059644443803045, LogSumExp(new DoubleVector([1, 2, 3])));
        Assert.Equal(1000 + Math.Log(2), LogSumExp(new DoubleVector([1000, 1000])));
        Assert.Equal(-1000 + Math.Log(2), LogSumExp(new DoubleVector([-1000, -1000])));
        AssertClose(Math.Log(1000), LogSumExp(new DoubleVector(1000)));
        Assert.Equal(double.NegativeInfinity, LogSumExp(new DoubleVector(0)));
        Assert.Equal(double.NegativeInfinity, LogSumExp(new DoubleVector([double.NegativeInfinity, double.NegativeInfinity])));
        Assert.Equal(1.0, LogSumExp(new DoubleVector([double.NegativeInfinity, 1])));
        Assert.Equal(double.PositiveInfinity, LogSumExp(new DoubleVector([1, double.PositiveInfinity, double.NegativeInfinity])));
        Assert.True(double.IsNaN(LogSumExp(new DoubleVector([1, double.NaN, double.PositiveInfinity]))));
        AssertClose(3.4076059644443803045, LogSumExp(new[] { 1f, 2f, 3f }.ToVector()), 1e-6);

        var a = Matrix<double>.FromColumnMajor(5, 70, RandomValues(350, 17));
        for (int j = 0; j < 70; j++)
        {
            a[1, j] = double.NegativeInfinity;
        }

        a[2, 5] = double.PositiveInfinity;
        a[3, 7] = double.NaN;
        a[4, 0] = 800;
        var rows = RowLogSumExp(a);
        var columns = ColumnLogSumExp(a);
        for (int i = 0; i < a.Rows; i++)
        {
            AssertSameBits(LogSumExp(a.Row(i)), rows[i]);
        }

        for (int j = 0; j < a.Columns; j++)
        {
            AssertSameBits(LogSumExp(a.Column(j)), columns[j]);
        }

        Assert.Equal(double.NegativeInfinity, rows[1]);
        Assert.Equal(double.PositiveInfinity, rows[2]);
        Assert.True(double.IsNaN(rows[3]));
        Assert.Equal(800.0, rows[4]);
        Assert.Equal(new[] { double.NegativeInfinity, double.NegativeInfinity }, RowLogSumExp(new Matrix<double>(2, 0)).ToArray());
        Assert.Equal(new[] { double.NegativeInfinity, double.NegativeInfinity }, ColumnLogSumExp(new Matrix<double>(0, 2)).ToArray());
        var destination = new DoubleVector(5);
        Assert.Same(destination, RowLogSumExp(a, destination));
        Assert.Throws<ArgumentException>(() => { ColumnLogSumExp(a, destination); });
    }

    [Fact]
    public void LogSoftmaxIsTheLogarithmOfSoftmaxWithoutLosingDigits()
    {
        // (x - max) - ln Σ exp(x - max)。最大値の項は和に含めず、ln(1 + ·) を log1p で求める
        var x = new DoubleVector([1, 2, 3]);
        double logSum = Log1p(Math.Exp(-2) + Math.Exp(-1));
        Assert.Equal(new[] { -2 - logSum, -1 - logSum, -logSum }, LogSoftmax(x).ToArray());
        var softmax = Softmax(x);
        var exp = LogSoftmax(x).ToArray().Select(Math.Exp).ToArray();
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(softmax[i], exp[i], 1e-15);
        }

        // ソフトマックスが 0 に丸められる要素でも、対数はそのまま残る
        Assert.Equal(new[] { 0.0, -1000 }, LogSoftmax(new DoubleVector([0, -1000])).ToArray());
        Assert.Equal(0.0, Softmax(new DoubleVector([0, -1000]))[1]);
        Assert.Equal(0, LogSoftmax(new DoubleVector(0)).Length);
        Assert.All(LogSoftmax(new DoubleVector([1, double.NaN])).ToArray(), v => Assert.True(double.IsNaN(v)));

        var destination = new DoubleVector(3);
        Assert.Same(destination, LogSoftmax(x, destination));
        var expected = LogSoftmax(x).ToArray();
        InPlace.LogSoftmax(x);
        Assert.Equal(expected, x.ToArray());

        var a = Matrix<double>.FromColumnMajor(3, 70, RandomValues(210, 19));
        var columns = ColumnLogSoftmax(a);
        var rows = RowLogSoftmax(a);
        for (int j = 0; j < a.Columns; j++)
        {
            Assert.Equal(LogSoftmax(a.Column(j)).ToArray(), columns.Column(j).ToArray());
        }

        for (int i = 0; i < a.Rows; i++)
        {
            Assert.Equal(LogSoftmax(a.Row(i)).ToArray(), rows.Row(i).ToArray());
        }

        Assert.Equal(rows.ToArray(), ColumnLogSoftmax(a.T).T.ToArray());
        var matrixDestination = new Matrix<double>(3, 70);
        Assert.Same(matrixDestination, RowLogSoftmax(a, matrixDestination));
        Assert.Same(matrixDestination, ColumnLogSoftmax(a, matrixDestination));
        var b = a.Clone();
        InPlace.RowLogSoftmax(b);
        Assert.Equal(rows.ToArray(), b.ToArray());
        InPlace.ColumnLogSoftmax(a);
        Assert.Equal(columns.ToArray(), a.ToArray());
        Assert.Equal(0, RowLogSoftmax(new Matrix<double>(0, 3)).Rows);
        Assert.Throws<ArgumentException>(() => { RowLogSoftmax(b, new Matrix<double>(70, 3)); });
        Assert.Throws<ArgumentException>(() => { ColumnLogSoftmax(b[.., ..3], b[.., 1..4]); });
    }

    // 差 x - max が正確に表せる平行移動では、LogSoftmax の結果はビット単位で不変であることを確認する（参照値は mpmath 50 桁）。
    [Fact]
    public void LogSoftmaxIsInvariantUnderShifts()
    {
        double[] small = [0, 1, 2, 8];
        var expected = LogSoftmax(new DoubleVector(small)).ToArray();
        AssertClose(-8.00371917206763, expected[0]);
        foreach (double shift in new[] { 1e9, -1e9, 1e15, -1e15 })
        {
            var shifted = new DoubleVector(small.Select(v => v + shift).ToArray());
            Assert.Equal(expected, LogSoftmax(shifted).ToArray());
            var rows = Matrix<double>.FromColumnMajor(1, 4, shifted.ToArray());
            Assert.Equal(expected, RowLogSoftmax(rows).Row(0).ToArray());
            Assert.Equal(expected, ColumnLogSoftmax(rows.T).Column(0).ToArray());
        }

        double[] offsets = [0.25, -1.5, 3];
        AssertClose(-2.8223549472662373, LogSoftmax(new DoubleVector(offsets.Select(v => v + 1e9).ToArray()))[0]);
    }

    // 最大の項が支配的で結果が 0 に近いときでも、Log1p により相対精度を保つことを確認する（参照値は mpmath 50 桁）。
    [Fact]
    public void LogSumExpKeepsRelativeAccuracyNearZero()
    {
        AssertClose(4.248354255291589e-18, LogSumExp(new DoubleVector([0, -40])));
        AssertClose(1.8625185805493936e-20, LogSumExp(new DoubleVector([-1e-20, -45])), 1e-14);
        AssertClose(-4.248354255291589e-18, LogSoftmax(new DoubleVector([10, -30]))[0]);
        AssertClose(-4.248354255291589e-18, RowLogSoftmax(Matrix<double>.FromColumnMajor(1, 2, [10, -30]))[0, 0]);
        AssertClose(4.248354255291589e-18, RowLogSumExp(Matrix<double>.FromColumnMajor(1, 2, [0, -40]))[0]);
        Assert.Equal(1000 + Math.Log(3), LogSumExp(new DoubleVector([1000, 1000, 1000])));
    }

    // Half のノルム・LogSumExp・ソフトマックスでは要素数が多いと Half のままでは和が 65504 を超えてオーバーフローするため、
    // float に拡幅して計算し最後に Half へ丸めることを検証する（double 参照値に対して 1 ulp 以内）。
    [Fact]
    public void HalfNormsAndNormalizationsKeepLongSumsInSingle()
    {
        static void Check(string api, int row, int index, double expected, Half actual)
        {
            double rounded = Math.Abs((double)(Half)expected);
            double ulp = (double)Half.BitIncrement((Half)rounded) - rounded;
            if (!Half.IsFinite(actual) || Math.Abs((double)actual - expected) > ulp)
            {
                Assert.Fail($"{api} row {row} [{index}]: expected {expected:R}, actual {(double)actual:R}");
            }
        }

        // 1行目は 0 に近い値ばかりで Σ exp(x - max) ≈ n、2行目は [1, 2) に縮めた平方和が約 4n/3 になる。3行目は幅の広い値
        const int n = 70000;
        var random = new Random(55);
        var values = new Half[3 * n];
        for (int j = 0; j < n; j++)
        {
            values[3 * j] = (Half)(-0.01 * random.NextDouble());
            values[3 * j + 1] = (Half)(2 * random.NextDouble() - 1);
            values[3 * j + 2] = (Half)(6 * random.NextDouble() - 3);
        }

        var a = Matrix<Half>.FromColumnMajor(3, n, values);
        var rowNorms = RowNorms(a);
        var columnNorms = ColumnNorms(a.T);
        var rowLogSumExp = RowLogSumExp(a);
        var columnLogSumExp = ColumnLogSumExp(a.T);
        var rowSoftmax = RowSoftmax(a);
        var columnSoftmax = ColumnSoftmax(a.T);
        var rowLogSoftmax = RowLogSoftmax(a);
        var columnLogSoftmax = ColumnLogSoftmax(a.T);
        var inPlace = a.Clone();
        InPlace.RowSoftmax(inPlace);
        for (int i = 0; i < 3; i++)
        {
            double[] x = Enumerable.Range(0, n).Select(j => (double)a[i, j]).ToArray();
            double max = x.Max();
            double logSum = Math.Log(x.Sum(v => Math.Exp(v - max)));
            double norm = Math.Sqrt(x.Sum(v => v * v));
            Check("Norm", i, 0, norm, Norm(a.Row(i)));
            Check("RowNorms", i, 0, norm, rowNorms[i]);
            Check("ColumnNorms", i, 0, norm, columnNorms[i]);
            Check("LogSumExp", i, 0, max + logSum, LogSumExp(a.Row(i)));
            Check("RowLogSumExp", i, 0, max + logSum, rowLogSumExp[i]);
            Check("ColumnLogSumExp", i, 0, max + logSum, columnLogSumExp[i]);
            var softmax = Softmax(a.Row(i));
            var logSoftmax = LogSoftmax(a.Row(i));
            var inPlaceLog = a.Row(i).ToVector();
            InPlace.LogSoftmax(inPlaceLog);
            for (int j = 0; j < n; j++)
            {
                double probability = Math.Exp(x[j] - max - logSum), logProbability = (x[j] - max) - logSum;
                Check("Softmax", i, j, probability, softmax[j]);
                Check("RowSoftmax", i, j, probability, rowSoftmax[i, j]);
                Check("ColumnSoftmax", i, j, probability, columnSoftmax[j, i]);
                Check("InPlace.RowSoftmax", i, j, probability, inPlace[i, j]);
                Check("LogSoftmax", i, j, logProbability, logSoftmax[j]);
                Check("RowLogSoftmax", i, j, logProbability, rowLogSoftmax[i, j]);
                Check("ColumnLogSoftmax", i, j, logProbability, columnLogSoftmax[j, i]);
                Check("InPlace.LogSoftmax", i, j, logProbability, inPlaceLog[j]);
            }
        }
    }

    // 1 つだけ 1 で残り 199,999 個が 0 のロジットは Σ exp(x - max) が Half の最大値を超える。
    // layout: 0 = ベクトル、1 = 1 行の行列（行ごとの API）、2 = 1 列の行列（列ごとの API）。double 参照値に対して 2 ulp 以内。
    private const int HalfLogitCount = 200000;

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void HalfSoftmaxRetainsRepresentableProbabilities(int layout)
    {
        Half[] actual = HalfNormalize(HalfLogits(), layout, logarithm: false);
        double denominator = Math.Exp(1) + HalfLogitCount - 1;
        AssertHalfWithinTwoUlps((Half)(Math.Exp(1) / denominator), actual[0]);
        AssertHalfWithinTwoUlps((Half)(1 / denominator), actual[^1]);
        // Half への個別の丸めは許すが、確率の和が 0 になるのは誤り
        double sum = 0;
        foreach (Half value in actual)
        {
            sum += (double)value;
        }

        Assert.InRange(sum, 0.99, 1.01);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void HalfLogSoftmaxRetainsFiniteLogProbabilities(int layout)
    {
        Half[] actual = HalfNormalize(HalfLogits(), layout, logarithm: true);
        double logDenominator = Math.Log(Math.Exp(1) + HalfLogitCount - 1);
        AssertHalfWithinTwoUlps((Half)(1 - logDenominator), actual[0]);
        AssertHalfWithinTwoUlps((Half)(-logDenominator), actual[^1]);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void HalfLogSumExpRemainsFinite(int layout)
    {
        Half[] input = HalfLogits();
        var matrix = HalfAsMatrix(input, layout);
        Half actual = layout switch
        {
            0 => XMath.LogSumExp<Half>(new ReadOnlyVectorSpan<Half>(input)),
            1 => XMath.RowLogSumExp<Half>(matrix)[0],
            _ => XMath.ColumnLogSumExp<Half>(matrix)[0],
        };
        AssertHalfWithinTwoUlps((Half)Math.Log(Math.Exp(1) + HalfLogitCount - 1), actual);
    }

    // 100,000 個の 1 の平方和は Half の最大値を超えるが、ノルム √100000 は表せる
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void HalfNormRemainsFiniteWhenSquaredNormOverflows(int layout)
    {
        Half[] input = Enumerable.Repeat((Half)1, 100000).ToArray();
        var matrix = HalfAsMatrix(input, layout);
        Half actual = layout switch
        {
            0 => XMath.Norm<Half>(new ReadOnlyVectorSpan<Half>(input)),
            1 => XMath.RowNorms<Half>(matrix)[0],
            _ => XMath.ColumnNorms<Half>(matrix)[0],
        };
        AssertHalfWithinTwoUlps((Half)Math.Sqrt(input.Length), actual);
    }

    private static Half[] HalfLogits()
    {
        var input = new Half[HalfLogitCount];
        input[0] = (Half)1;
        return input;
    }

    private static ReadOnlyMatrixSpan<Half> HalfAsMatrix(Half[] input, int layout) => layout == 1
        ? new(input, new nint[] { 1, input.Length }, new nint[] { 1, 1 })
        : new(input, new nint[] { input.Length, 1 }, new nint[] { 1, input.Length });

    private static Half[] HalfNormalize(Half[] input, int layout, bool logarithm)
    {
        if (layout == 0)
        {
            var vector = new ReadOnlyVectorSpan<Half>(input);
            return (logarithm ? XMath.LogSoftmax<Half>(vector) : XMath.Softmax<Half>(vector)).AsSpan().ToArray();
        }

        var matrix = HalfAsMatrix(input, layout);
        return (layout == 1
            ? logarithm ? XMath.RowLogSoftmax<Half>(matrix) : XMath.RowSoftmax<Half>(matrix)
            : logarithm ? XMath.ColumnLogSoftmax<Half>(matrix) : XMath.ColumnSoftmax<Half>(matrix)).AsSpan().ToArray();
    }

    private static void AssertHalfWithinTwoUlps(Half expected, Half actual)
    {
        double tolerance = 2 * Math.Abs((double)Half.BitIncrement(expected) - (double)expected);
        Assert.True(Half.IsFinite(actual) && Math.Abs((double)actual - (double)expected) <= tolerance,
            $"expected={(double)expected:R}, actual={(double)actual:R}, tolerance={tolerance:R}");
    }

    // Half の縮約・ソフトマックス・分散は、要素ごとに (float) で広げて float の同じ API で求め、(Half) で丸めた値とビット単位で同じ。
    // 広げる・丸めるのは SIMD 版（HalfConversion）なので、ベクトルの幅の前後の長さ、非連続のビュー、その場更新、特殊値で比べる。
    // NaN の内容は保証しないので、NaN どうしは同じとみなす
    [Fact]
    public void HalfReductionsMatchTheSinglePrecisionPathBitwise()
    {
        static ushort Bits(Half x) => Half.IsNaN(x) ? (ushort)0x7E00 : BitConverter.HalfToUInt16Bits(x);
        static void Same(string api, ReadOnlySpan<Half> expected, ReadOnlySpan<Half> actual)
        {
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                if (Bits(expected[i]) != Bits(actual[i]))
                {
                    Assert.Fail($"{api} [{i}]: expected {expected[i]}, actual {actual[i]}");
                }
            }
        }

        static Matrix<float> Widened(ReadOnlyMatrixSpan<Half> a)
        {
            var result = new Matrix<float>(a.Rows, a.Columns);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Columns; j++)
                {
                    result[i, j] = (float)a[i, j];
                }
            }

            return result;
        }

        static Half[] Narrowed(ReadOnlyMatrixSpan<float> a)
        {
            var result = new Half[a.Rows * a.Columns];
            for (int j = 0; j < a.Columns; j++)
            {
                for (int i = 0; i < a.Rows; i++)
                {
                    result[j * a.Rows + i] = (Half)a[i, j];
                }
            }

            return result;
        }

        static Half[] Each(int count, Func<int, float> reduce) => Enumerable.Range(0, count).Select(i => (Half)reduce(i)).ToArray();

        static void Check(string view, ReadOnlyMatrixSpan<Half> a)
        {
            int rows = a.Rows, columns = a.Columns;
            var w = Widened(a);
            Same($"{view} RowNorms", Each(rows, i => Norm(w.Row(i).ToVector())), RowNorms(a).AsSpan());
            Same($"{view} ColumnNorms", Each(columns, j => Norm(w.Column(j))), ColumnNorms(a).AsSpan());
            Same($"{view} RowLogSumExp", Each(rows, i => LogSumExp(w.Row(i).ToVector())), RowLogSumExp(a).AsSpan());
            Same($"{view} ColumnLogSumExp", Each(columns, j => LogSumExp(w.Column(j))), ColumnLogSumExp(a).AsSpan());
            Same($"{view} RowVariances", Each(rows, i => Variance(w.Row(i).ToVector(), 1)), RowVariances(a, 1).AsSpan());
            Same($"{view} RowStandardDeviations", Each(rows, i => StandardDeviation(w.Row(i).ToVector(), 0)), RowStandardDeviations(a, 0).AsSpan());
            Same($"{view} ColumnVariances", Each(columns, j => Variance(w.Column(j), 1)), ColumnVariances(a, 1).AsSpan());

            // ソフトマックス系は行列全体を列優先の float に写し、その場で求めてから丸める
            foreach (string api in new[] { "RowSoftmax", "ColumnSoftmax", "RowLogSoftmax", "ColumnLogSoftmax" })
            {
                var expected = w.Clone();
                var actual = new Matrix<Half>(rows, columns);
                var padded = new Matrix<Half>(rows + 2, columns + 3);
                var target = padded[1..(rows + 1), 2..(columns + 2)];
                a.CopyTo(target);
                switch (api)
                {
                    case "RowSoftmax": InPlace.RowSoftmax(expected); RowSoftmax(a, actual); InPlace.RowSoftmax(target); break;
                    case "ColumnSoftmax": InPlace.ColumnSoftmax(expected); ColumnSoftmax(a, actual); InPlace.ColumnSoftmax(target); break;
                    case "RowLogSoftmax": InPlace.RowLogSoftmax(expected); RowLogSoftmax(a, actual); InPlace.RowLogSoftmax(target); break;
                    default: InPlace.ColumnLogSoftmax(expected); ColumnLogSoftmax(a, actual); InPlace.ColumnLogSoftmax(target); break;
                }

                var narrowed = Narrowed(expected);
                Same($"{view} {api}", narrowed, actual.AsSpan());
                Same($"{view} InPlace.{api}", narrowed, target.ToMatrix().AsSpan());
            }

            // 出力先が非連続（転置のビュー）
            var transposed = new Matrix<Half>(columns, rows);
            ColumnLogSoftmax(a, transposed.T);
            Same($"{view} ColumnLogSoftmax->T", ColumnLogSoftmax(a).AsSpan(), transposed.T.ToMatrix().AsSpan());

            if (rows > 0 && columns > 0)
            {
                var v = a.Column(0);
                var wv = w.Column(0).ToVector();
                Same($"{view} Norm", [(Half)Norm(wv)], [Norm(v)]);
                Same($"{view} LogSumExp", [(Half)LogSumExp(wv)], [LogSumExp(v)]);
                Same($"{view} Variance", [(Half)Variance(wv, 0)], [Variance(v, 0)]);
                var softmax = wv.Clone();
                InPlace.Softmax(softmax);
                Same($"{view} Softmax", Narrowed(softmax.AsMatrix(rows, 1)), Softmax(v).AsSpan());
                var logSoftmax = wv.Clone();
                InPlace.LogSoftmax(logSoftmax);
                Same($"{view} LogSoftmax", Narrowed(logSoftmax.AsMatrix(rows, 1)), LogSoftmax(v).AsSpan());
            }
        }

        var random = new Random(56);
        Half[] specials = [Half.NaN, Half.PositiveInfinity, Half.NegativeInfinity, Half.Zero, Half.NegativeZero, Half.Epsilon, -Half.Epsilon,
            Half.MaxValue, Half.MinValue, BitConverter.UInt16BitsToHalf(0x03FF), BitConverter.UInt16BitsToHalf(0x0400)];
        foreach (int rows in new[] { 1, 7, 33, 65, 130 })
        {
            foreach (int columns in new[] { 1, 2, 17, 64 })
            {
                foreach (int kind in new[] { 0, 1, 2 })
                {
                    // 0: [-3, 3)、1: 全ビット列（NaN・∞を含む）、2: 小さな値と特殊値
                    Half Value() => kind switch
                    {
                        0 => (Half)(6 * random.NextDouble() - 3),
                        1 => BitConverter.UInt16BitsToHalf((ushort)random.Next(65536)),
                        _ => random.Next(10) == 0 ? specials[random.Next(specials.Length)] : (Half)((2 * random.NextDouble() - 1) * 1e-3),
                    };

                    var values = Enumerable.Range(0, (rows + 2) * (columns + 2)).Select(_ => Value()).ToArray();
                    var padded = Matrix<Half>.FromColumnMajor(rows + 2, columns + 2, values);
                    string shape = $"{rows}x{columns} kind {kind}";
                    Check($"{shape} dense", Matrix<Half>.FromColumnMajor(rows, columns, values[..(rows * columns)]));
                    Check($"{shape} sub", padded[1..(rows + 1), 1..(columns + 1)]);
                    Check($"{shape} transposed", Matrix<Half>.FromColumnMajor(columns, rows, values[..(rows * columns)]).T);
                }
            }
        }
    }

    // ---------- 初等関数 ----------

    [Fact]
    public void ElementaryFunctionsLiftTheScalarDefinitions()
    {
        var x = new DoubleVector([0.25, 1, 4]);
        var y = new DoubleVector([2, -1, 0.5]);
        Assert.Equal(x.ToArray().Select(v => Math.Log(v)).ToArray(), Log(x).ToArray());
        Assert.Equal(new[] { 0.5, 1, 2 }, Sqrt(x).ToArray());
        Assert.Equal(new[] { 2.0, 0, 3 }, Abs(new DoubleVector([-2, -0.0, 3])).ToArray());
        Assert.Equal(new[] { 0.0625, 1, 16 }, Square(x).ToArray());
        Assert.Equal(new[] { 4.0, 1, 0.25 }, Reciprocal(x).ToArray());
        Assert.Equal(new[] { 0.0625, 1, 16 }, Pow(x, 2.0).ToArray());
        Assert.Equal(new[] { 0.0625, 1, 2 }, Pow(x, y).ToArray());
        Assert.Equal(new[] { 0.125, 0.5, 2 }, Divide(x, 2.0).ToArray());
        Assert.Equal(new[] { 0.125, -1, 8 }, Divide(x, y).ToArray());
        Assert.Equal(new[] { 1.0, 1, 4 }, Max(x, 1.0).ToArray());
        Assert.Equal(new[] { 0.25, 1, 1 }, Min(x, 1.0).ToArray());
        Assert.Equal(new[] { 2.0, 1, 4 }, Max(x, y).ToArray());
        Assert.Equal(new[] { 0.25, -1, 0.5 }, Min(x, y).ToArray());
        Assert.Equal(new[] { 0.5, 1, 2 }, Clamp(x, 0.5, 2.0).ToArray());

        // 2引数の関数は NaN を伝え、符号つきゼロを区別する
        Assert.True(double.IsNaN(Max(new DoubleVector([double.NaN]), 0.0)[0]));
        Assert.True(double.IsNaN(Clamp(new DoubleVector([double.NaN]), 0.0, 1.0)[0]));
        Assert.True(double.IsPositive(Max(new DoubleVector([-0.0]), 0.0)[0]));
        Assert.True(double.IsNegative(Min(new DoubleVector([0.0]), -0.0)[0]));

        // min > max は要素があるときだけ例外
        Assert.Throws<ArgumentException>(() => { Clamp(x, 1.0, 0.0); });
        Assert.Equal(0, Clamp(new DoubleVector(0), 1.0, 0.0).Length);

        // 行列、転置ビュー、destination を受け取るオーバーロード、インプレース版
        var a = Matrix<double>.FromColumnMajor(2, 2, [1, 4, 9, 16]);
        Assert.Equal(new double[,] { { 1, 3 }, { 2, 4 } }, Sqrt(a).ToArray());
        Assert.Equal(Log(a).T.ToArray(), Log(a.T).ToArray());
        Assert.Equal(new double[,] { { 1, 1 }, { 1, 1 } }, Divide(a, a).ToArray());
        var destination = new Matrix<double>(2, 2);
        Assert.Same(destination, Pow(a, 0.5, destination));
        Assert.Equal(Sqrt(a).ToArray(), destination.ToArray());
        Assert.Same(destination, Clamp(a, 2.0, 10.0, destination));
        Assert.Equal(new double[,] { { 2, 9 }, { 4, 10 } }, destination.ToArray());
        InPlace.Sqrt(a);
        InPlace.Pow(a, 2.0);
        InPlace.Max(a, 4.0);
        Assert.Equal(new double[,] { { 4, 9 }, { 4, 16 } }, a.ToArray());
        var z = new DoubleVector([-2, 0.5, 3]);
        InPlace.Clamp(z, -1.0, 1.0);
        InPlace.Divide(z, 2.0);
        InPlace.Abs(z);
        Assert.Equal(new[] { 0.5, 0.25, 0.5 }, z.ToArray());
        Assert.Throws<ArgumentException>(() => { Divide(x, new DoubleVector(2)); });

        // 演算子
        Assert.Equal(Sqrt(x).ToArray(), Map<XMath.Operators.Sqrt<double>>(x).ToArray());
        Assert.Equal(Pow(x, 3.0).ToArray(), Map(x, new XMath.Operators.PowScalar<double>(3.0)).ToArray());
        Assert.Equal(Pow(x, y).ToArray(), Zip<XMath.Operators.Pow<double>>(x, y).ToArray());
        Assert.Equal(Clamp(x, 0.5, 2.0).ToArray(), Map(x, new XMath.Operators.Clamp<double>(0.5, 2.0)).ToArray());
        Assert.Equal(Min(x, y).ToArray(), Zip<XMath.Operators.Min<double>>(x, y).ToArray());

        // スカラー・整数・複素数
        Assert.Equal(9.0, Square(3.0));
        Assert.Equal(0.25, Reciprocal(4.0));
        Assert.Equal(1.0, Clamp(5.0, 0.0, 1.0));
        Assert.Equal(3.0, Max(3.0, 2.0));
        Assert.Equal(new[] { 3, 2 }, Abs(new[] { -3, 2 }.ToVector()).ToArray());
        Assert.Equal(new[] { 9, 16 }, Square(new[] { 3, -4 }.ToVector()).ToArray());
        Assert.Equal(new[] { 3, -3 }, Divide(new[] { 7, -7 }.ToVector(), 2).ToArray());
        Assert.Equal(new[] { 3, 5 }, Max(new[] { 1, 5 }.ToVector(), 3).ToArray());
        Assert.Equal(new[] { -1, 0, 1 }, Clamp(new[] { -5, 0, 5 }.ToVector(), -1, 1).ToArray());
        Assert.Equal(new[] { 8.0, 9 }, Pow(new[] { 2.0, 3 }.ToVector(), new[] { 3.0, 2 }.ToVector()).ToArray());
        Assert.Equal(new Complex(5, 0), Abs(new[] { new Complex(3, 4) }.ToVector())[0]);
        Assert.Equal(new Complex(0, 2), Square(new[] { new Complex(1, 1) }.ToVector())[0]);
    }

    [Fact]
    public void PreciseFunctionsMatchHighPrecisionReferences()
    {
        // 参照値は mpmath（40桁）で、double に丸めた入力から計算した
        Assert.Equal(1e-20, Log1p(1e-20));
        Assert.Equal(0.0, Math.Log(1 + 1e-20));                          // 素直に計算すると 0 になる
        Assert.Equal(1e-20, Expm1(1e-20));
        AssertClose(9.8596765437597708567e-305, Logistic(-700.0));
        AssertClose(0.95257412682243321912, Logistic(3.0));
        AssertClose(4.0000003309614839964e-10, Logit(0.5000000001), 1e-14);
        AssertClose(-69.077552789821370437, Logit(1e-30));
        AssertClose(20.794415415867036707, Logit(1 - Math.Pow(2, -30)));
        AssertClose(-4.2483542552915889863e-18, LogLogistic(40.0));
        AssertClose(-3.0485873515737420588, LogLogistic(-3.0));
        AssertClose(-46.05170185988091368, Log1mExp(-1e-20));
        AssertClose(-1.928749847963917783e-22, Log1mExp(-50.0));
        AssertClose(-0.93275212956718857189, Log1mExp(-0.5));

        // 端と範囲外
        Assert.Equal(0.5, Logistic(0.0));
        Assert.Equal(1.0, Logistic(800.0));
        Assert.Equal(0.0, Logistic(-800.0));
        Assert.Equal(1.0, Logistic(double.PositiveInfinity));
        Assert.Equal(0.0, Logistic(double.NegativeInfinity));
        Assert.Equal(0.0, Logit(0.5));
        Assert.Equal(double.NegativeInfinity, Logit(0.0));
        Assert.Equal(double.PositiveInfinity, Logit(1.0));
        Assert.True(double.IsNaN(Logit(1.5)));
        Assert.True(double.IsNaN(Logit(-0.1)));
        AssertClose(2.0, Logit(Logistic(2.0)), 1e-14);
        Assert.Equal(-1000.0, LogLogistic(-1000.0));
        Assert.Equal(0.0, LogLogistic(1000.0));
        Assert.Equal(double.NegativeInfinity, Log1mExp(0.0));
        Assert.True(double.IsNaN(Log1mExp(1.0)));
        foreach (double v in new[] { Log1p(double.NaN), Expm1(double.NaN), Logistic(double.NaN), Logit(double.NaN), LogLogistic(double.NaN), Log1mExp(double.NaN) })
        {
            Assert.True(double.IsNaN(v));
        }

        // ベクトル・行列・単精度・演算子
        double[] points = [-30, -1, -1e-12, 0, 1e-12, 0.5, 3, 30];
        var x = points.ToVector();
        Assert.Equal(points.Select(Logistic).ToArray(), Logistic(x).ToArray());
        Assert.Equal(points.Select(LogLogistic).ToArray(), LogLogistic(x).ToArray());
        Assert.Equal(points.Select(Log1p).ToArray(), Log1p(x).ToArray());
        Assert.Equal(points.Select(Expm1).ToArray(), Expm1(x).ToArray());
        var p = new DoubleVector([1e-30, 0.25, 0.5000000001, 0.75, 1 - 1e-10]);
        Assert.Equal(p.ToArray().Select(Logit).ToArray(), Logit(p).ToArray());
        var negative = new DoubleVector([-1e-20, -0.5, -50]);
        Assert.Equal(negative.ToArray().Select(Log1mExp).ToArray(), Log1mExp(negative).ToArray());
        var a = Matrix<double>.FromColumnMajor(2, 2, [-1, 0, 1, 2]);
        Assert.Equal(Logistic(a).T.ToArray(), Logistic(a.T).ToArray());
        var expected = LogLogistic(a).ToArray();
        InPlace.LogLogistic(a);
        Assert.Equal(expected, a.ToArray());
        Assert.Equal(1e-20f, Log1p(new[] { 1e-20f }.ToVector())[0]);
        Assert.Equal((float)Logistic(-80.0), Logistic(new[] { -80f }.ToVector())[0]);
        Assert.Equal(Logit(p).ToArray(), Map<XMath.Operators.Logit<double>>(p).ToArray());
    }

    [Fact]
    public void LogSpaceFunctionsMatchHighPrecisionReferences()
    {
        // 参照値は mpmath（入力は double をそのまま使い、打ち消しに足りる桁数で計算した）
        AssertClose(-4.999999999666667031e-21, Log1pmx(1e-10));
        AssertClose(-0.056674943938732374155, Log1pmx(-0.3));
        AssertClose(-0.091223880042632224123, Log1pmx(0.49));
        AssertClose(-0.094534891891835618022, Log1pmx(0.5));
        AssertClose(-0.24289607400945110912, Log1pmx(0.8674929218566181));
        AssertClose(-5.9087552789821361648, Log1pmx(-0.999));
        AssertClose(-1.6137056388801093812, Log1pmx(3.0));
        AssertClose(-9999999976.97414907, Log1pmx(1e10));
        AssertClose(-0.32174375051815135289, Log1pmx(-0.6036080605175855));

        AssertClose(0.69314718055994530942, LogAddExp(0.0, 0.0));
        AssertClose(-999.30685281944005469, LogAddExp(-1000.0, -1000.0));             // eˣ は 0 に、
        AssertClose(1000.3132616875182228, LogAddExp(1000.0, 999.0));                 // あるいは ∞ になる
        AssertClose(1000.3132616875182228, LogAddExp(999.0, 1000.0));
        AssertClose(0.00180929401297573714, LogAddExp(0.001809294009765758, -26.46294738336573));
        AssertClose(-744.68673831248177717, LogAddExp(-745.0, -746.0));
        Assert.Equal(30.0, LogAddExp(30.0, -30.0));
        AssertClose(-0.00023920985723999660402, LogAddExp(-0.6926878384926739, -0.6940854306576218), 1e-12);   // 条件数が約 3000

        AssertClose(-46.051701859880913735, LogSubExp(0.0, -1e-20));
        AssertClose(-1.928749847963917783e-22, LogSubExp(0.0, -50.0));
        AssertClose(999.54132485461291811, LogSubExp(1000.0, 999.0));
        AssertClose(-0.86722780561545897782, LogSubExp(0.0006985219441528122, -0.5437204577577232));
        AssertClose(-745.45867514538708189, LogSubExp(-745.0, -746.0));
        AssertClose(-21.025850847250089268, LogSubExp(2.0, 1.9999999999));            // 差 y - x は丸めなしで求まる

        // 端と範囲外
        Assert.Equal(0.0, Log1pmx(0.0));
        Assert.Equal(double.NegativeInfinity, Log1pmx(-1.0));
        Assert.Equal(double.NegativeInfinity, Log1pmx(double.PositiveInfinity));
        Assert.True(double.IsNaN(Log1pmx(-2.0)));
        Assert.True(double.IsNaN(Log1pmx(double.NegativeInfinity)));
        Assert.Equal(double.NegativeInfinity, LogAddExp(double.NegativeInfinity, double.NegativeInfinity));
        Assert.Equal(-3.0, LogAddExp(double.NegativeInfinity, -3.0));
        Assert.Equal(double.PositiveInfinity, LogAddExp(double.PositiveInfinity, double.PositiveInfinity));
        Assert.Equal(double.PositiveInfinity, LogAddExp(double.NegativeInfinity, double.PositiveInfinity));
        Assert.Equal(double.NegativeInfinity, LogSubExp(1.5, 1.5));
        Assert.Equal(double.NegativeInfinity, LogSubExp(double.NegativeInfinity, double.NegativeInfinity));
        Assert.Equal(-3.0, LogSubExp(-3.0, double.NegativeInfinity));
        Assert.Equal(double.PositiveInfinity, LogSubExp(double.PositiveInfinity, 0.0));
        Assert.True(double.IsNaN(LogSubExp(0.0, 1.0)));
        Assert.True(double.IsNaN(LogSubExp(double.PositiveInfinity, double.PositiveInfinity)));
        foreach (double v in new[]
        {
            Log1pmx(double.NaN), LogAddExp(double.NaN, 0.0), LogAddExp(0.0, double.NaN), LogAddExp(double.PositiveInfinity, double.NaN),
            LogAddExp(double.NaN, double.NegativeInfinity), LogSubExp(double.NaN, 0.0), LogSubExp(0.0, double.NaN), LogSubExp(double.PositiveInfinity, double.NaN),
        })
        {
            Assert.True(double.IsNaN(v));
        }

        // 対称性と、e^LogSubExp(LogAddExp(x, y), y) = eˣ
        Assert.Equal(LogAddExp(-2.5, 7.25), LogAddExp(7.25, -2.5));
        AssertClose(-2.5, LogSubExp(LogAddExp(-2.5, -1.0), -1.0), 1e-14);

        // ベクトル・行列・単精度・演算子
        double[] xs = [-1, -0.5, -1e-12, 0, 1e-12, 0.25, 3, 1e6];
        double[] ys = [-2, 0.5, -40, 0, 1e-12, -0.25, 3, -1e6];
        var x = xs.ToVector();
        var y = ys.ToVector();
        Assert.Equal(xs.Select(v => Log1pmx(v)).ToArray(), Log1pmx(x).ToArray());
        Assert.Equal(xs.Zip(ys, (u, v) => LogAddExp(u, v)).ToArray(), LogAddExp(x, y).ToArray());
        Assert.Equal(xs.Select(v => LogAddExp(v, 0.5)).ToArray(), LogAddExp(x, 0.5).ToArray());
        Assert.Equal(xs.Select(v => LogSubExp(v, -1.5)).ToArray(), LogSubExp(x, -1.5).ToArray());
        double[] big = [0, 1, -1e-12, 5];
        double[] small = [-1, 0.5, -3e-12, double.NegativeInfinity];
        Assert.Equal(big.Zip(small, (u, v) => LogSubExp(u, v)).ToArray(), LogSubExp(big.ToVector(), small.ToVector()).ToArray());
        var a = Matrix<double>.FromColumnMajor(2, 2, [-1, 0, 1, 2]);
        var b = Matrix<double>.FromColumnMajor(2, 2, [-3, -2, 0.5, 1]);
        Assert.Equal(LogAddExp(a, b).T.ToArray(), LogAddExp(a.T, b.T).ToArray());
        var destination = new Matrix<double>(2, 2);
        Assert.Same(destination, LogSubExp(a, b, destination));
        Assert.Equal(new[,] { { LogSubExp(-1.0, -3.0), LogSubExp(1.0, 0.5) }, { LogSubExp(0.0, -2.0), LogSubExp(2.0, 1.0) } }, destination.ToArray());
        var expected = Log1pmx(a).ToArray();
        InPlace.Log1pmx(a);
        Assert.Equal(expected, a.ToArray());
        Assert.Equal((float)LogAddExp(-80.0, -81.0), LogAddExp(new[] { -80f }.ToVector(), -81f)[0]);
        Assert.Equal((float)Log1pmx((double)1e-4f), Log1pmx(new[] { 1e-4f }.ToVector())[0]);
        Assert.Equal(LogAddExp(x, y).ToArray(), Zip<XMath.Operators.LogAddExp<double>>(x, y).ToArray());
        Assert.Equal(LogSubExp(x, -1.5).ToArray(), Map(x, new XMath.Operators.LogSubExpScalar<double>(-1.5)).ToArray());
        Assert.Equal(Log1pmx(x).ToArray(), Map<XMath.Operators.Log1pmx<double>>(x).ToArray());
    }
}
