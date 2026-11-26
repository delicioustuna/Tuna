using Tuna.LinearAlgebra;
using static Tuna.Tests.LinearAlgebra.LinearAlgebraTestUtilities;
using DenseVector = Tuna.LinearAlgebra.Vector<double>;

namespace Tuna.Tests.LinearAlgebra;

// 関数で要素を決める行列の生成: 添字の演算（Generate・GenerateBand・GenerateSymmetric）と、
// 2点の演算（Pairwise・PairwiseSymmetric）、既定の距離の演算（SquaredDistance・Distance・ScaledSquaredDistance）。
// 並列の経路とスレッド数によらない結果は MatrixGenerationParallelTests で確かめる。
[Trait("LinearAlgebra", "Generation")]
public class MatrixGenerationTests
{
    internal readonly struct Hilbert : IIndexOperator<double>
    {
        public static double Invoke(int row, int column) => 1.0 / (row + column + 1);
    }

    internal readonly struct HilbertSingle : IIndexOperator<float>
    {
        public static float Invoke(int row, int column) => 1f / (row + column + 1);
    }

    /// <summary>1次元の線形要素の剛性行列（三重対角 [−1, 2, −1]）</summary>
    internal readonly struct Laplacian : IIndexOperator<double>
    {
        public static double Invoke(int row, int column) => row == column ? 2 : -1;
    }

    internal readonly struct Toeplitz(double[] c) : IStatefulIndexOperator<double>
    {
        public double Invoke(int row, int column) => c[Math.Abs(row - column)];
    }

    /// <summary>帯 <c>−lower ≤ j − i ≤ upper</c> の外を呼んだら投げ、要素ごとの呼び出し回数を数える</summary>
    internal readonly struct BandChecked(int rows, int lower, int upper, int[] calls) : IStatefulIndexOperator<double>
    {
        public double Invoke(int row, int column)
        {
            if (column - row > upper || row - column > lower)
            {
                throw new InvalidOperationException($"({row}, {column}) is outside the band.");
            }

            Interlocked.Increment(ref calls[row + column * rows]);
            return Value(row, column);
        }

        public static double Value(int row, int column) => 1 + row + 1000.0 * column;
    }

    internal readonly struct ThrowsAt(int row, int column) : IStatefulIndexOperator<double>
    {
        public double Invoke(int i, int j) => i == row && j == column ? throw new InvalidOperationException("boom") : i + j;
    }

    internal readonly struct ThrowingPair : IPairwiseOperator<double>
    {
        public static double Invoke(ReadOnlyVectorSpan<double> x, ReadOnlyVectorSpan<double> y) =>
            x[0] == y[0] ? throw new InvalidOperationException("boom") : 0;
    }

    private static Matrix<double> Filled(int rows, int columns, double value = double.NaN)
    {
        var m = new Matrix<double>(rows, columns);
        m.AsSpan().Fill(value);
        return m;
    }

    private static double NaiveSquaredDistance(Matrix<double> x, int i, Matrix<double> y, int j)
    {
        double sum = 0;
        for (int k = 0; k < x.Rows; k++)
        {
            double d = x[k, i] - y[k, j];
            sum += d * d;
        }

        return sum;
    }

    // ---- 添字の演算 ----

    [Fact]
    public void GenerateWritesEveryElementFromTheIndices()
    {
        var h = Matrix<double>.Generate<Hilbert>(4, 3);
        Assert.Equal(new double[,] { { 1, 1 / 2.0, 1 / 3.0 }, { 1 / 2.0, 1 / 3.0, 1 / 4.0 }, { 1 / 3.0, 1 / 4.0, 1 / 5.0 }, { 1 / 4.0, 1 / 5.0, 1 / 6.0 } },
            h.ToArray());

        var destination = Filled(4, 3);
        Assert.Equal(h.ToArray(), XMath.Generate<Hilbert>(destination).ToMatrix().ToArray());

        var t = Matrix<double>.Generate(3, 4, new Toeplitz([5, 1, 0.5, 0.25]));
        Assert.Equal(new double[,] { { 5, 1, 0.5, 0.25 }, { 1, 5, 1, 0.5 }, { 0.5, 1, 5, 1 } }, t.ToArray());
        var transposed = Filled(4, 3);
        XMath.Generate(new Toeplitz([5, 1, 0.5, 0.25]), transposed.T);
        Assert.Equal(t.ToArray(), XMath.Transpose(transposed).ToArray());

        Assert.Equal(new float[,] { { 1, 0.5f }, { 0.5f, 1 / 3f } }, Matrix<float>.Generate<HilbertSingle>(2, 2).ToArray());
        Assert.Equal(0, Matrix<double>.Generate<Hilbert>(0, 5).Rows);
    }

    [Theory]
    [InlineData(7, 7, 1, 1)]
    [InlineData(7, 7, 0, 0)]
    [InlineData(9, 5, 2, 1)]
    [InlineData(5, 9, 0, 3)]
    [InlineData(6, 6, 0, 5)]
    [InlineData(6, 4, 10, 10)]
    [InlineData(1, 1, 0, 0)]
    public void GenerateBandEvaluatesTheBandOnceAndZerosTheRest(int rows, int columns, int lower, int upper)
    {
        var calls = new int[rows * columns];
        var destination = Filled(rows + 2, columns + 1);
        var view = destination[1..(rows + 1), 1..];
        XMath.GenerateBand(lower, upper, new BandChecked(rows, lower, upper, calls), view);
        for (int j = 0; j < columns; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                bool inside = j - i <= upper && i - j <= lower;
                Assert.Equal(inside ? BandChecked.Value(i, j) : 0, view[i, j]);
                Assert.Equal(inside ? 1 : 0, calls[i + j * rows]);
            }
        }

        // 範囲の外は書かない
        Assert.True(double.IsNaN(destination[0, 0]));
        Assert.True(double.IsNaN(destination[rows + 1, columns]));

        var allocated = Matrix<double>.GenerateBand(rows, columns, lower, upper, new BandChecked(rows, lower, upper, new int[rows * columns]));
        Assert.Equal(view.ToMatrix().ToArray(), allocated.ToArray());
    }

    [Theory]
    [InlineData(1, 0)]
    [InlineData(8, 0)]
    [InlineData(8, 1)]
    [InlineData(8, 3)]
    [InlineData(8, 7)]
    [InlineData(8, 20)]
    public void GenerateSymmetricEvaluatesOnlyTheLowerHalfOfTheBand(int n, int bandwidth)
    {
        var calls = new int[n * n];
        var destination = Filled(n, n);
        XMath.GenerateSymmetric(bandwidth, new BandChecked(n, bandwidth, 0, calls), destination);
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                bool inside = Math.Abs(i - j) <= bandwidth;
                double expected = inside ? BandChecked.Value(Math.Max(i, j), Math.Min(i, j)) : 0;
                Assert.Equal(expected, destination[i, j]);
                Assert.Equal(inside && i >= j ? 1 : 0, calls[i + j * n]);
            }
        }

        // 帯幅を指定しない版は全体（下三角を求めて写す）
        var full = Filled(n, n);
        XMath.GenerateSymmetric(new BandChecked(n, n, 0, new int[n * n]), full);
        Assert.Equal(full.ToArray(), Matrix<double>.GenerateSymmetric(n, new BandChecked(n, n, 0, new int[n * n])).ToArray());
        Assert.Equal(full.ToArray(), XMath.Transpose(full).ToArray());
    }

    [Fact]
    public void GenerateSymmetricBuildsABandedStiffnessMatrix()
    {
        int n = 50;
        var k = Matrix<double>.GenerateSymmetric<Laplacian>(n, 1);
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                Assert.Equal(i == j ? 2 : Math.Abs(i - j) == 1 ? -1 : 0, k[i, j]);
            }
        }

        // 両端を固定した棒: 一様な荷重の変位は放物線 u_i = (i + 1)(n − i)/2
        var f = new DenseVector(n);
        f.AsSpan().Fill(1);
        var u = k.Cholesky().Solve(f);
        for (int i = 0; i < n; i++)
        {
            Assert.Equal((i + 1) * (n - i) / 2.0, u[i], 1e-10);
        }

        var destination = Filled(n, n);
        XMath.GenerateSymmetric<Laplacian>(1, destination);
        Assert.Equal(k.ToArray(), destination.ToArray());
        Assert.Equal(Matrix<double>.GenerateBand<Laplacian>(n, n, 1, 1).ToArray(), k.ToArray());
        Assert.Equal(XMath.GenerateBand<Laplacian>(1, 1, Filled(n, n)).ToMatrix().ToArray(), k.ToArray());
    }

    [Fact]
    public void GenerateRejectsInvalidShapesAndBandwidths()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => { XMath.GenerateBand<Hilbert>(-1, 0, new Matrix<double>(3, 3)); });
        Assert.Throws<ArgumentOutOfRangeException>(() => { XMath.GenerateBand<Hilbert>(0, -1, new Matrix<double>(3, 3)); });
        Assert.Throws<ArgumentOutOfRangeException>(() => { XMath.GenerateSymmetric<Hilbert>(-1, new Matrix<double>(3, 3)); });
        Assert.Throws<ArgumentException>(() => { XMath.GenerateSymmetric<Hilbert>(new Matrix<double>(3, 4)); });
        Assert.Throws<ArgumentException>(() => { XMath.GenerateSymmetric<Hilbert>(1, new Matrix<double>(4, 3)); });
        Assert.Throws<ArgumentOutOfRangeException>(() => { Matrix<double>.GenerateBand<Hilbert>(3, 3, -2, 0); });
        Assert.Throws<ArgumentOutOfRangeException>(() => { Matrix<double>.GenerateSymmetric<Hilbert>(3, -1); });
        Assert.Throws<ArgumentOutOfRangeException>(() => { Matrix<double>.Generate<Hilbert>(-1, 3); });
    }

    [Fact]
    public void ExceptionsFromTheOperatorPropagateUnchanged()
    {
        var e = Assert.Throws<InvalidOperationException>(() => { Matrix<double>.Generate(5, 5, new ThrowsAt(3, 2)); });
        Assert.Equal("boom", e.Message);
        Assert.Throws<InvalidOperationException>(() => { XMath.PairwiseSymmetric<ThrowingPair>(Filled(2, 3, 1)); });
    }

    // ---- 2点の演算 ----

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(64)]
    [InlineData(130)]
    public void PairwiseSymmetricMatchesPairwiseBitForBit(int n)
    {
        var x = Random(5, n, 200 + n);
        var expected = XMath.Pairwise<XMath.Operators.SquaredDistance<double>>(x, x);
        var symmetric = XMath.PairwiseSymmetric<XMath.Operators.SquaredDistance<double>>(x);
        Assert.Equal(expected.AsSpan().ToArray(), symmetric.AsSpan().ToArray());
        Assert.Equal(symmetric.ToArray(), XMath.Transpose(symmetric).ToArray());
        for (int j = 0; j < n; j++)
        {
            Assert.Equal(0, symmetric[j, j]);
            for (int i = 0; i < n; i++)
            {
                Assert.Equal(NaiveSquaredDistance(x, i, x, j), symmetric[i, j]);
            }
        }

        double[] scales = [0.5, 1.5, 2, 3, 0.25];
        var op = new XMath.Operators.ScaledSquaredDistance<double>(scales);
        Assert.Equal(XMath.Pairwise(x, x, op).AsSpan().ToArray(), XMath.PairwiseSymmetric(x, op).AsSpan().ToArray());
    }

    [Fact]
    public void PairwiseSymmetricWritesViewsAndRejectsBadDestinations()
    {
        int n = 9;
        var x = Random(3, n, 7);
        var expected = XMath.PairwiseSymmetric<XMath.Operators.Distance<double>>(x);

        var large = Filled(n + 2, n + 3, -1);
        XMath.PairwiseSymmetric<XMath.Operators.Distance<double>>(x, large[2.., 1..(n + 1)]);
        Assert.Equal(expected.ToArray(), large[2.., 1..(n + 1)].ToMatrix().ToArray());
        Assert.Equal(-1, large[0, 0]);
        Assert.Equal(-1, large[n + 1, n + 2]);

        var transposed = Filled(n, n);
        XMath.PairwiseSymmetric<XMath.Operators.Distance<double>>(x, transposed.T);
        Assert.Equal(expected.ToArray(), transposed.ToArray());

        // 点を行に並べた配置（X.T）でもよい
        var rows = XMath.Transpose(x);
        Assert.Equal(expected.ToArray(), XMath.PairwiseSymmetric<XMath.Operators.Distance<double>>(rows.T).ToArray());

        Assert.Throws<ArgumentException>(() => { XMath.PairwiseSymmetric<XMath.Operators.Distance<double>>(x, new Matrix<double>(n, n + 1)); });
        var overlapping = new Matrix<double>(n, n);
        Assert.Throws<ArgumentException>(() => { XMath.PairwiseSymmetric<XMath.Operators.Distance<double>>(overlapping[..3, ..], overlapping); });

        // 次元 0 の点はすべて同じ点
        Assert.Equal(new double[4, 4], XMath.PairwiseSymmetric<XMath.Operators.SquaredDistance<double>>(new Matrix<double>(0, 4)).ToArray());
    }

    [Fact]
    public void DistanceMatricesAreGeneratedFromTheTwoPointFunctions()
    {
        var x = Random(4, 6, 11);
        var y = Random(4, 5, 12);
        var squared = XMath.SquaredDistance(x, y);
        Assert.Equal(XMath.Pairwise<XMath.Operators.SquaredDistance<double>>(x, y).AsSpan().ToArray(), squared.AsSpan().ToArray());
        var distance = XMath.Distance(x, y);
        for (int j = 0; j < 5; j++)
        {
            for (int i = 0; i < 6; i++)
            {
                Assert.Equal(NaiveSquaredDistance(x, i, y, j), squared[i, j]);
                Assert.Equal(Math.Sqrt(squared[i, j]), distance[i, j]);
                Assert.Equal(distance[i, j], XMath.Distance(x.Column(i), y.Column(j)));
            }
        }

        var destination = Filled(6, 5);
        XMath.SquaredDistance(x, y, destination);
        Assert.Equal(squared.ToArray(), destination.ToArray());
        Assert.Throws<ArgumentException>(() => { XMath.SquaredDistance(x, Random(3, 5, 1)); });
        Assert.Throws<ArgumentException>(() => XMath.SquaredDistance(new DenseVector(2), new DenseVector(3)));
    }

    [Fact]
    public void DistanceRescalesWhenTheSquaredSumOverflowsOrUnderflows()
    {
        static double D(double[] a, double[] b) => XMath.Distance(new DenseVector(a), new DenseVector(b));

        Assert.Equal(5e200, D([3e200, 0], [0, 4e200]), 1e-15 * 5e200);
        Assert.Equal(Math.Sqrt(2) * 1e300, D([1e300, -1e300], [0, 0]), 1e-15 * 1e300);
        Assert.Equal(5e-200, D([3e-200, 0], [0, 4e-200]), 1e-15 * 5e-200);
        Assert.Equal(5 * double.Epsilon, D([3 * double.Epsilon, 0], [0, 4 * double.Epsilon]));
        Assert.Equal(0, D([1, 2], [1, 2]));
        Assert.Equal(double.PositiveInfinity, D([double.PositiveInfinity, 0], [0, 0]));
        Assert.Equal(double.PositiveInfinity, D([double.MaxValue, 0], [-double.MaxValue, 0]));
        Assert.True(double.IsNaN(D([double.NaN, 1e300], [0, 0])));
        Assert.True(double.IsNaN(D([double.PositiveInfinity], [double.PositiveInfinity])));
        Assert.Equal(25.0, XMath.SquaredDistance(new DenseVector([3.0, 0]), new DenseVector([0.0, 4])));
        Assert.Equal(5f, XMath.Distance(new Vector<float>([3f, 0]), new Vector<float>([0f, 4])));
    }

    [Fact]
    public void ScaledSquaredDistanceDividesEachAxisByItsLengthScale()
    {
        var x = Random(3, 8, 21);
        var y = Random(3, 6, 22);

        // ℓ = 2 なら 1/ℓ を掛けても丸めは同じ
        var uniform = XMath.Pairwise(x, y, new XMath.Operators.ScaledSquaredDistance<double>([2.0, 2.0, 2.0]));
        var squared = XMath.SquaredDistance(x, y);
        for (int j = 0; j < 6; j++)
        {
            for (int i = 0; i < 8; i++)
            {
                Assert.Equal(squared[i, j] / 4, uniform[i, j]);
            }
        }

        double[] scales = [0.3, 1.7, 4.1];
        var op = new XMath.Operators.ScaledSquaredDistance<double>(scales);
        Assert.Equal(3, op.Dimension);
        var scaled = XMath.Pairwise(x, y, op);
        for (int j = 0; j < 6; j++)
        {
            for (int i = 0; i < 8; i++)
            {
                double expected = 0;
                for (int k = 0; k < 3; k++)
                {
                    double d = (x[k, i] - y[k, j]) / scales[k];
                    expected += d * d;
                }

                Assert.Equal(expected, scaled[i, j], 1e-14 * expected);
            }
        }

        // 値はコピーする
        scales[0] = 100;
        Assert.Equal(scaled.ToArray(), XMath.Pairwise(x, y, op).ToArray());

        Assert.Throws<ArgumentException>(() => new XMath.Operators.ScaledSquaredDistance<double>([1.0, 0.0]));
        Assert.Throws<ArgumentException>(() => new XMath.Operators.ScaledSquaredDistance<double>([-1.0]));
        Assert.Throws<ArgumentException>(() => new XMath.Operators.ScaledSquaredDistance<double>([double.NaN]));
        Assert.Throws<ArgumentException>(() => new XMath.Operators.ScaledSquaredDistance<double>([double.PositiveInfinity]));
        Assert.Throws<ArgumentException>(() => { XMath.Pairwise(x, y, new XMath.Operators.ScaledSquaredDistance<double>([1.0, 1.0])); });
        Assert.Equal(0, default(XMath.Operators.ScaledSquaredDistance<double>).Dimension);
    }

    /// <summary>1/ℓ が溢れる・非正規化数になる尺度でも、同じ点は 0、尺度1つ分の差は 1（割って求める）</summary>
    [Fact]
    public void ScaledSquaredDistanceDividesWhenTheInverseScaleIsNotNormal()
    {
        static void Check<T>(T scale) where T : unmanaged, System.Numerics.IFloatingPointIeee754<T>
        {
            var op = new XMath.Operators.ScaledSquaredDistance<T>([scale, T.One]);
            var zero = new Vector<T>([T.Zero, T.Zero]);
            Assert.Equal(T.Zero, op.Invoke(zero, zero));
            Assert.Equal(T.One, op.Invoke(new Vector<T>([scale, T.Zero]), zero));
            Assert.Equal(T.One + T.One, op.Invoke(new Vector<T>([scale, T.One]), zero));

            // 距離行列の対角（自己距離）も 0
            var points = Matrix<T>.FromColumnMajor(2, 3, [T.Zero, T.Zero, scale, T.One, -scale, T.Zero]);
            var d = XMath.PairwiseSymmetric(points, op);
            for (int i = 0; i < 3; i++)
            {
                Assert.Equal(T.Zero, d[i, i]);
            }

            Assert.Equal(T.One, d[2, 0]);
        }

        Check((Half)Math.ScaleB(1, -20));      // 1/ℓ が溢れる
        Check((Half)32768);                    // 1/ℓ が非正規化数
        Check(MathF.ScaleB(1, -130));
        Check(MathF.ScaleB(1, 127));
        Check(Math.ScaleB(1, -1030));
        Check(Math.ScaleB(1, 1023));
    }

    // x=ℓ, y=-ℓ なら、正規化した差は常に2、二乗距離は厳密に4。
    // binary16/32/64の2冪尺度。中間のx-yだけが溢れる点と通常尺度を分けて検査する。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditIdentities20261102")]
    [Theory]
    [InlineData("double", 64.0, false)]
    [InlineData("double", 64.0, true)]
    [InlineData("double", 8.98846567431158e+307, false)]
    [InlineData("double", 8.98846567431158e+307, true)]
    [InlineData("float", 64.0, false)]
    [InlineData("float", 64.0, true)]
    [InlineData("float", 1.7014118346046923e+38, false)]
    [InlineData("float", 1.7014118346046923e+38, true)]
    [InlineData("Half", 64.0, false)]
    [InlineData("Half", 64.0, true)]
    [InlineData("Half", 32768.0, false)]
    [InlineData("Half", 32768.0, true)]
    public void ScaledSquaredDistanceNormalizesBeforeAnOverflowingDifference(string type, double scale, bool pairwise)
    {
        void Check<T>(T lengthScale) where T : unmanaged, System.Numerics.IFloatingPointIeee754<T>
        {
            var op = new XMath.Operators.ScaledSquaredDistance<T>([lengthScale]);
            T actual;
            if (pairwise)
            {
                var points = Matrix<T>.FromColumnMajor(1, 2, [lengthScale, -lengthScale]);
                actual = XMath.PairwiseSymmetric(points, op)[1, 0];
            }
            else
            {
                actual = op.Invoke(new Vector<T>([lengthScale]), new Vector<T>([-lengthScale]));
            }

            Assert.Equal(T.CreateChecked(4), actual);
        }

        switch (type)
        {
            case "double": Check(scale); break;
            case "float": Check((float)scale); break;
            case "Half": Check((Half)scale); break;
            default: throw new ArgumentException(nameof(type));
        }
    }
}
