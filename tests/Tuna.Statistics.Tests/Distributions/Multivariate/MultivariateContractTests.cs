using System.Runtime.InteropServices;
using Tuna.LinearAlgebra;
using Tuna.RandomSource;
using Tuna.Statistics.Distributions;

namespace Tuna.Tests.Statistics;

// 多変量分布・行列値の分布に共通の規約（IMultivariateContinuousDistribution・IMultivariateDiscreteDistribution・IMatrixContinuousDistribution の説明）。
// 評価点の NaN・±∞・台の外の扱い、形状の検証、複数の点・標本の版が1点・1標本の版の繰り返しと一致すること、
// ストライドのあるビューを受け付けること、パラメータを複製して持つことを、すべての分布で確かめる。
public class MultivariateContractTests
{
    internal static readonly double[] Covariance3 = [2.0, 0.3, -0.4, 0.3, 1.0, 0.2, -0.4, 0.2, 1.5];
    internal static readonly double[] Mean3 = [1.0, -2.0, 0.5];

    internal static ReadOnlyMatrixSpan<double> Square(double[] values) => new(values, (int)Math.Sqrt(values.Length), (int)Math.Sqrt(values.Length));

    internal static IMultivariateContinuousDistribution Vector(string name) => name switch
    {
        "MultivariateNormal" => new MultivariateNormalDistribution(Mean3, Square(Covariance3)),
        "MultivariateNormalFromFactor" => MultivariateNormalDistribution.FromCholeskyFactor(Mean3, Square([1.5, 0.2, -0.3, 0, 0.8, 0.1, 0, 0, 1.2])),
        "MultivariateStudentT" => new MultivariateStudentTDistribution(3.5, Mean3, Square(Covariance3)),
        "MultivariateCauchy" => new MultivariateStudentTDistribution(1, Mean3, Square(Covariance3)),
        "MultivariateStudentTSmall" => new MultivariateStudentTDistribution(0.4, Mean3, Square(Covariance3)),
        "MultivariateStudentTInfinite" => new MultivariateStudentTDistribution(double.PositiveInfinity, Mean3, Square(Covariance3)),
        "Dirichlet" => new DirichletDistribution(2, 3, 0.5),
        "DirichletSmall" => new DirichletDistribution(0.01, 0.02, 0.05),
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    internal static IMatrixContinuousDistribution Matrix(string name) => name switch
    {
        "Wishart" => new WishartDistribution(4.5, Square(Covariance3)),
        "WishartSmallDegrees" => new WishartDistribution(2.2, Square(Covariance3)),
        "InverseWishart" => new InverseWishartDistribution(6.5, Square(Covariance3)),
        "MatrixNormal" => new MatrixNormalDistribution(
            new ReadOnlyMatrixSpan<double>([1.0, 0.0, -1.0, 0.5, 2.0, 0.25], 3, 2), Square(Covariance3), Square([2.0, 0.3, 0.3, 0.5])),
        "Lkj" => new LkjCorrelationDistribution(4, 1.5),
        "LkjSmall" => new LkjCorrelationDistribution(3, 0.3),
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    public static TheoryData<string> VectorNames =>
    [
        "MultivariateNormal", "MultivariateNormalFromFactor", "MultivariateStudentT", "MultivariateCauchy", "MultivariateStudentTSmall",
        "MultivariateStudentTInfinite", "Dirichlet", "DirichletSmall",
    ];

    public static TheoryData<string> MatrixNames => ["Wishart", "WishartSmallDegrees", "InverseWishart", "MatrixNormal", "Lkj", "LkjSmall"];

    internal static IMultivariateDiscreteDistribution Discrete(string name) => name switch
    {
        "Multinomial" => new MultinomialDistribution(12, 0.2, 0.3, 0.5),
        "MultinomialZeroProbability" => new MultinomialDistribution(9, 0.0, 0.25, 0.0, 0.75),
        "MultinomialLarge" => new MultinomialDistribution(1L << 40, 0.1, 0.6, 0.3),
        "MultinomialSingle" => new MultinomialDistribution(7, 1.0),
        "DirichletMultinomial" => new DirichletMultinomialDistribution(12, 2, 3, 0.5),
        "DirichletMultinomialSmall" => new DirichletMultinomialDistribution(30, 0.01, 0.02, 0.05),
        "NegativeMultinomial" => new NegativeMultinomialDistribution(2.5, 0.4, 0.25, 0.35),
        "NegativeMultinomialZeroProbability" => new NegativeMultinomialDistribution(0.7, 0.2, 0.0, 0.8, 0.0),
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    public static TheoryData<string> DiscreteNames =>
    [
        "Multinomial", "MultinomialZeroProbability", "MultinomialLarge", "MultinomialSingle", "DirichletMultinomial", "DirichletMultinomialSmall",
        "NegativeMultinomial", "NegativeMultinomialZeroProbability",
    ];

    private static bool IsSymmetricValued(IMatrixContinuousDistribution d) => d is not MatrixNormalDistribution;

    /// <summary>転置を新しい行列に書く（その <c>.T</c> は元と同じ値で、行のストライドが1でないビューになる）</summary>
    internal static Matrix<double> Transposed(Matrix<double> x)
    {
        var result = new Matrix<double>(x.Columns, x.Rows);
        for (int i = 0; i < x.Rows; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                result[j, i] = x[i, j];
            }
        }

        return result;
    }

    // ---------------------------------------------------------------- ベクトル値の分布

    /// <summary>分布の標本と、台の外・NaN を含む列を並べた評価点（d × 12）</summary>
    private static Matrix<double> VectorPoints(IMultivariateContinuousDistribution d)
    {
        int n = d.Dimension;
        var points = new Matrix<double>(n, 12);
        d.Sample(new Pcg64DxsmRandom(1, 2), points[.., ..8]);
        points.Column(8).Fill(double.NaN);
        points.Column(9).Fill(0.25);
        points[0, 9] = double.PositiveInfinity;
        points.Column(10).Fill(0.25);
        points[n - 1, 10] = double.NegativeInfinity;
        points.Column(11).Fill(-0.5);
        return points;
    }

    [Theory]
    [MemberData(nameof(VectorNames))]
    public void VectorPdfIsTheExponentialOfLogPdf(string name)
    {
        var d = Vector(name);
        var points = VectorPoints(d);
        for (int j = 0; j < points.Columns; j++)
        {
            double logPdf = d.LogPdf(points.Column(j));
            Assert.Equal(Math.Exp(logPdf), d.Pdf(points.Column(j)));
        }
    }

    [Theory]
    [MemberData(nameof(VectorNames))]
    public void VectorEvaluationPointsWithNaNOrInfinity(string name)
    {
        var d = Vector(name);
        var points = VectorPoints(d);
        Assert.True(double.IsNaN(d.LogPdf(points.Column(8))));
        Assert.True(double.IsNaN(d.Pdf(points.Column(8))));
        Assert.Equal(double.NegativeInfinity, d.LogPdf(points.Column(9)));
        Assert.Equal(double.NegativeInfinity, d.LogPdf(points.Column(10)));
        Assert.Equal(0.0, d.Pdf(points.Column(9)));

        // NaN は他の成分が ±∞ でも NaN
        double[] mixed = new double[d.Dimension];
        mixed[0] = double.PositiveInfinity;
        mixed[^1] = double.NaN;
        Assert.True(double.IsNaN(d.LogPdf(mixed)));
    }

    [Theory]
    [MemberData(nameof(VectorNames))]
    public void VectorBulkLogPdfMatchesRepeatedSinglePoints(string name)
    {
        var d = Vector(name);
        var points = VectorPoints(d);
        var bulk = d.LogPdf(points);
        Assert.Equal(points.Columns, bulk.Length);
        for (int j = 0; j < points.Columns; j++)
        {
            Assert.Equal(d.LogPdf(points.Column(j)), bulk[j]);
        }

        // 転置して持った行列の転置ビュー（行のストライドが1でない）と、ストライドのある書き込み先
        var transposed = Transposed(points);
        var destination = new Matrix<double>(2, points.Columns);
        d.LogPdf(transposed.T, destination.Row(1));
        Assert.Equal(bulk.ToArray(), destination.Row(1).ToArray());
    }

    [Theory]
    [MemberData(nameof(VectorNames))]
    public void VectorShapeMismatchThrows(string name)
    {
        var d = Vector(name);
        int n = d.Dimension;
        var random = new Pcg64DxsmRandom(1, 2);
        Assert.Throws<ArgumentException>(() => d.LogPdf(new double[n + 1]));
        Assert.Throws<ArgumentException>(() => d.Pdf(new double[n - 1]));
        Assert.Throws<ArgumentException>(() => d.LogPdf(new Matrix<double>(n + 1, 2), new double[2]));
        Assert.Throws<ArgumentException>(() => d.LogPdf(new Matrix<double>(n, 2), new double[3]));
        Assert.Throws<ArgumentException>(() => d.Sample(random, new double[n - 1]));
        Assert.Throws<ArgumentException>(() => d.Sample(random, new Matrix<double>(n + 1, 2)));
    }

    [Theory]
    [MemberData(nameof(VectorNames))]
    public void VectorDestinationOverlappingThePointsThrows(string name)
    {
        var d = Vector(name);
        int n = d.Dimension;
        double[] storage = new double[n * 4 + 4];
        storage.AsSpan().Fill(0.25);
        Assert.Throws<ArgumentException>(() => d.LogPdf(new ReadOnlyMatrixSpan<double>(storage.AsSpan(0, n * 4), n, 4), storage.AsSpan(n * 4 - 2, 4)));
        d.LogPdf(new ReadOnlyMatrixSpan<double>(storage.AsSpan(0, n * 4), n, 4), storage.AsSpan(n * 4, 4));
    }

    [Theory]
    [MemberData(nameof(VectorNames))]
    public void VectorNullArgumentsThrow(string name)
    {
        var d = Vector(name);
        Assert.Throws<ArgumentNullException>(() => d.Sample(null!, new double[d.Dimension]));
        Assert.Throws<ArgumentNullException>(() => d.Sample(null!, new Matrix<double>(d.Dimension, 2)));
        Assert.Throws<ArgumentNullException>(() => d.Sample(null!));
        Assert.Throws<ArgumentNullException>(() => d.Sample(null!, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => d.Sample(new Pcg64DxsmRandom(1, 2), -1));
        IMultivariateContinuousDistribution missing = null!;
        Assert.Throws<ArgumentNullException>(() => missing.Sample(new Pcg64DxsmRandom(1, 2)));
        Assert.Throws<ArgumentNullException>(() => missing.LogPdf(new Matrix<double>(3, 2)));
    }

    [Theory]
    [MemberData(nameof(VectorNames))]
    public void VectorBulkSamplingMatchesRepeatedSingleSamples(string name)
    {
        var d = Vector(name);
        var single = new Philox4x32Random(7, 3);
        var bulk = new Philox4x32Random(7, 3);
        var strided = new Philox4x32Random(7, 3);
        var values = d.Sample(bulk, 33);
        Assert.Equal(d.Dimension, values.Rows);
        Assert.Equal(33, values.Columns);
        for (int j = 0; j < values.Columns; j++)
        {
            Assert.Equal(d.Sample(single).ToArray(), values.Column(j).ToArray());
        }

        Assert.Equal(single.NextUInt64(), bulk.NextUInt64());

        // 書き込み先が転置ビュー（列のストライドが1でない）でも同じ値
        var transposed = new Matrix<double>(33, d.Dimension);
        d.Sample(strided, transposed.T);
        for (int j = 0; j < values.Columns; j++)
        {
            Assert.Equal(values.Column(j).ToArray(), transposed.Row(j).ToArray());
        }

        Assert.Empty(d.Sample(single, 0).AsSpan().ToArray());
    }

    [Theory]
    [MemberData(nameof(VectorNames))]
    public void VectorSamplesLieInTheSupport(string name)
    {
        var d = Vector(name);
        var samples = d.Sample(new Pcg64DxsmRandom(20261028, 5), 2000);
        var logPdf = d.LogPdf(samples);
        Assert.All(samples.AsSpan().ToArray(), v => Assert.True(double.IsFinite(v)));
        Assert.All(logPdf.ToArray(), v => Assert.True(v > double.NegativeInfinity, $"log density {v}"));
    }

    [Theory]
    [MemberData(nameof(VectorNames))]
    public void VectorMomentsAreFreshCopies(string name)
    {
        var d = Vector(name);
        var mean = d.GetMean();
        var covariance = d.GetCovariance();
        double[] expectedMean = mean.ToArray();
        double[] expectedCovariance = covariance.AsSpan().ToArray();
        mean.AsSpan().Fill(42);
        covariance.AsSpan().Fill(42);
        Assert.Equal(expectedMean, d.GetMean().ToArray());
        Assert.Equal(expectedCovariance, d.GetCovariance().AsSpan().ToArray());
        Assert.Equal(d.Dimension, expectedMean.Length);
        Assert.Equal(d.Dimension * d.Dimension, expectedCovariance.Length);
    }

    // ---------------------------------------------------------------- 離散の多変量分布

    /// <summary>分布の標本と、台の外の列（負の成分、総数の合わない点、long の最大値）を並べた評価点（d × 12）</summary>
    private static Matrix<long> DiscretePoints(IMultivariateDiscreteDistribution d)
    {
        int n = d.Dimension;
        var points = new Matrix<long>(n, 12);
        d.Sample(new Pcg64DxsmRandom(1, 2), points[.., ..8]);
        points.Column(8).Fill(1);
        points[0, 8] = -1;
        points.Column(9).Fill(long.MaxValue);
        points.Column(10).Fill(long.MinValue);
        points.Column(11).Fill(0);
        points[n - 1, 11] = 1;
        return points;
    }

    private static bool HasFixedTotal(IMultivariateDiscreteDistribution d) => d is not NegativeMultinomialDistribution;

    [Theory]
    [MemberData(nameof(DiscreteNames))]
    public void DiscretePmfIsTheExponentialOfLogPmf(string name)
    {
        var d = Discrete(name);
        var points = DiscretePoints(d);
        for (int j = 0; j < points.Columns; j++)
        {
            double logPmf = d.LogPmf(points.Column(j));
            Assert.False(double.IsNaN(logPmf), $"column {j}");
            Assert.True(logPmf <= 0, $"column {j}: {logPmf}");
            Assert.Equal(Math.Exp(logPmf), d.Pmf(points.Column(j)));
        }
    }

    [Theory]
    [MemberData(nameof(DiscreteNames))]
    public void DiscretePointsOutsideTheSupportHaveZeroProbability(string name)
    {
        var d = Discrete(name);
        var points = DiscretePoints(d);
        Assert.Equal(double.NegativeInfinity, d.LogPmf(points.Column(8)));
        Assert.Equal(0.0, d.Pmf(points.Column(8)));
        Assert.Equal(double.NegativeInfinity, d.LogPmf(points.Column(10)));
        if (HasFixedTotal(d))
        {
            // 成分の和が試行回数と違う点（long の最大値を並べても溢れずに台の外と判定する）
            Assert.Equal(double.NegativeInfinity, d.LogPmf(points.Column(9)));
            long[] shifted = d.Sample(new Pcg64DxsmRandom(3, 4)).ToArray();
            shifted[0]++;
            Assert.Equal(double.NegativeInfinity, d.LogPmf(shifted));
        }
    }

    [Theory]
    [MemberData(nameof(DiscreteNames))]
    public void DiscreteBulkLogPmfMatchesRepeatedSinglePoints(string name)
    {
        var d = Discrete(name);
        var points = DiscretePoints(d);
        var bulk = d.LogPmf(points);
        Assert.Equal(points.Columns, bulk.Length);
        for (int j = 0; j < points.Columns; j++)
        {
            Assert.Equal(d.LogPmf(points.Column(j)), bulk[j]);
        }

        // 転置して持った行列の転置ビュー（行のストライドが1でない）と、ストライドのある書き込み先
        var transposed = new Matrix<long>(points.Columns, points.Rows);
        for (int i = 0; i < points.Rows; i++)
        {
            for (int j = 0; j < points.Columns; j++)
            {
                transposed[j, i] = points[i, j];
            }
        }

        var destination = new Matrix<double>(2, points.Columns);
        d.LogPmf(transposed.T, destination.Row(1));
        Assert.Equal(bulk.ToArray(), destination.Row(1).ToArray());
    }

    [Theory]
    [MemberData(nameof(DiscreteNames))]
    public void DiscreteShapeMismatchThrows(string name)
    {
        var d = Discrete(name);
        int n = d.Dimension;
        var random = new Pcg64DxsmRandom(1, 2);
        Assert.Throws<ArgumentException>(() => d.LogPmf(new long[n + 1]));
        Assert.Throws<ArgumentException>(() => d.Pmf(new long[n + 2]));
        Assert.Throws<ArgumentException>(() => d.LogPmf(new Matrix<long>(n + 1, 2), new double[2]));
        Assert.Throws<ArgumentException>(() => d.LogPmf(new Matrix<long>(n, 2), new double[3]));
        Assert.Throws<ArgumentException>(() => d.Sample(random, new long[n + 1]));
        Assert.Throws<ArgumentException>(() => d.Sample(random, new Matrix<long>(n + 1, 2)));
    }

    [Theory]
    [MemberData(nameof(DiscreteNames))]
    public void DiscreteDestinationOverlappingThePointsThrows(string name)
    {
        // 評価点（long）と書き込み先（double）が同じメモリを共有する場合
        var d = Discrete(name);
        int n = d.Dimension;
        long[] storage = new long[n * 4 + 4];
        var points = new ReadOnlyMatrixSpan<long>(storage.AsSpan(0, n * 4), n, 4);
        Assert.Throws<ArgumentException>(() => d.LogPmf(
            new ReadOnlyMatrixSpan<long>(storage.AsSpan(0, n * 4), n, 4), MemoryMarshal.Cast<long, double>(storage.AsSpan(n * 4 - 2, 4))));
        d.LogPmf(points, MemoryMarshal.Cast<long, double>(storage.AsSpan(n * 4, 4)));
    }

    [Theory]
    [MemberData(nameof(DiscreteNames))]
    public void DiscreteNullArgumentsThrow(string name)
    {
        var d = Discrete(name);
        Assert.Throws<ArgumentNullException>(() => d.Sample(null!, new long[d.Dimension]));
        Assert.Throws<ArgumentNullException>(() => d.Sample(null!, new Matrix<long>(d.Dimension, 2)));
        Assert.Throws<ArgumentNullException>(() => d.Sample(null!));
        Assert.Throws<ArgumentNullException>(() => d.Sample(null!, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => d.Sample(new Pcg64DxsmRandom(1, 2), -1));
        IMultivariateDiscreteDistribution missing = null!;
        Assert.Throws<ArgumentNullException>(() => missing.Sample(new Pcg64DxsmRandom(1, 2)));
        Assert.Throws<ArgumentNullException>(() => missing.LogPmf(new Matrix<long>(3, 2)));
    }

    [Theory]
    [MemberData(nameof(DiscreteNames))]
    public void DiscreteBulkSamplingMatchesRepeatedSingleSamples(string name)
    {
        var d = Discrete(name);
        var single = new Philox4x32Random(7, 3);
        var bulk = new Philox4x32Random(7, 3);
        var strided = new Philox4x32Random(7, 3);
        var values = d.Sample(bulk, 33);
        Assert.Equal(d.Dimension, values.Rows);
        Assert.Equal(33, values.Columns);
        for (int j = 0; j < values.Columns; j++)
        {
            Assert.Equal(d.Sample(single).ToArray(), values.Column(j).ToArray());
        }

        Assert.Equal(single.NextUInt64(), bulk.NextUInt64());

        // 書き込み先が転置ビュー（列のストライドが1でない）でも同じ値
        var transposed = new Matrix<long>(33, d.Dimension);
        d.Sample(strided, transposed.T);
        for (int j = 0; j < values.Columns; j++)
        {
            Assert.Equal(values.Column(j).ToArray(), transposed.Row(j).ToArray());
        }

        Assert.Empty(d.Sample(single, 0).AsSpan().ToArray());
    }

    [Theory]
    [MemberData(nameof(DiscreteNames))]
    public void DiscreteSamplesLieInTheSupport(string name)
    {
        var d = Discrete(name);
        var samples = d.Sample(new Pcg64DxsmRandom(20261029, 5), 2000);
        var logPmf = d.LogPmf(samples);
        Assert.All(samples.AsSpan().ToArray(), v => Assert.True(v >= 0, $"{v}"));
        Assert.All(logPmf.ToArray(), v => Assert.True(v > double.NegativeInfinity && v <= 0, $"log probability {v}"));
    }

    [Theory]
    [MemberData(nameof(DiscreteNames))]
    public void DiscreteMomentsAreFreshCopies(string name)
    {
        var d = Discrete(name);
        var mean = d.GetMean();
        var covariance = d.GetCovariance();
        double[] expectedMean = mean.ToArray();
        double[] expectedCovariance = covariance.AsSpan().ToArray();
        mean.AsSpan().Fill(42);
        covariance.AsSpan().Fill(42);
        Assert.Equal(expectedMean, d.GetMean().ToArray());
        Assert.Equal(expectedCovariance, d.GetCovariance().AsSpan().ToArray());
        Assert.Equal(d.Dimension, expectedMean.Length);
        Assert.Equal(d.Dimension * d.Dimension, expectedCovariance.Length);
        for (int i = 0; i < d.Dimension; i++)
        {
            for (int j = 0; j < i; j++)
            {
                Assert.Equal(covariance[i, j], covariance[j, i]);
            }
        }
    }

    // ---------------------------------------------------------------- 行列値の分布

    private static Matrix<double> MatrixPoint(IMatrixContinuousDistribution d, int seed) => d.Sample(new Pcg64DxsmRandom((ulong)seed, 2));

    [Theory]
    [MemberData(nameof(MatrixNames))]
    public void MatrixPdfIsTheExponentialOfLogPdfAndAcceptsStridedViews(string name)
    {
        var d = Matrix(name);
        for (int seed = 0; seed < 8; seed++)
        {
            var x = MatrixPoint(d, seed);
            double logPdf = d.LogPdf(x);
            Assert.True(double.IsFinite(logPdf), $"log density {logPdf}");
            Assert.Equal(Math.Exp(logPdf), d.Pdf(x));

            Assert.Equal(logPdf, d.LogPdf(Transposed(x).T));
        }
    }

    [Theory]
    [MemberData(nameof(MatrixNames))]
    public void MatrixEvaluationPointsWithNaNOrInfinity(string name)
    {
        var d = Matrix(name);
        var x = MatrixPoint(d, 1);
        x[d.Rows - 1, 0] = double.NaN;
        Assert.True(double.IsNaN(d.LogPdf(x)));
        Assert.True(double.IsNaN(d.Pdf(x)));
        x[0, d.Columns - 1] = double.PositiveInfinity;
        Assert.True(double.IsNaN(d.LogPdf(x)));

        x = MatrixPoint(d, 1);
        x[0, 0] = double.PositiveInfinity;
        Assert.Equal(double.NegativeInfinity, d.LogPdf(x));
        Assert.Equal(0.0, d.Pdf(x));
        x[0, 0] = double.NegativeInfinity;
        Assert.Equal(double.NegativeInfinity, d.LogPdf(x));
    }

    [Theory]
    [MemberData(nameof(MatrixNames))]
    public void MatrixShapeMismatchThrows(string name)
    {
        var d = Matrix(name);
        var random = new Pcg64DxsmRandom(1, 2);
        Assert.Throws<ArgumentException>(() => d.LogPdf(new Matrix<double>(d.Rows + 1, d.Columns)));
        Assert.Throws<ArgumentException>(() => d.Pdf(new Matrix<double>(d.Rows, d.Columns + 1)));
        Assert.Throws<ArgumentException>(() => d.Sample(random, new Matrix<double>(d.Rows, d.Columns - 1)));
        Assert.Throws<ArgumentException>(() => d.Sample(random, new Matrix<double>(d.Rows - 1, d.Columns)));
    }

    [Theory]
    [MemberData(nameof(MatrixNames))]
    public void MatrixNullArgumentsThrow(string name)
    {
        var d = Matrix(name);
        Assert.Throws<ArgumentNullException>(() => d.Sample(null!, new Matrix<double>(d.Rows, d.Columns)));
        Assert.Throws<ArgumentNullException>(() => d.Sample(null!));
        IMatrixContinuousDistribution missing = null!;
        Assert.Throws<ArgumentNullException>(() => missing.Sample(new Pcg64DxsmRandom(1, 2)));
    }

    [Theory]
    [MemberData(nameof(MatrixNames))]
    public void MatrixSamplingIntoStridedViewsMatchesContiguousDestination(string name)
    {
        var d = Matrix(name);
        var contiguous = new Philox4x32Random(7, 3);
        var strided = new Philox4x32Random(7, 3);
        for (int k = 0; k < 5; k++)
        {
            var expected = d.Sample(contiguous);
            var transposed = new Matrix<double>(d.Columns, d.Rows);
            d.Sample(strided, transposed.T);
            for (int i = 0; i < d.Rows; i++)
            {
                for (int j = 0; j < d.Columns; j++)
                {
                    Assert.Equal(expected[i, j], transposed[j, i]);
                }
            }
        }

        Assert.Equal(contiguous.NextUInt64(), strided.NextUInt64());
    }

    [Theory]
    [MemberData(nameof(MatrixNames))]
    public void MatrixSamplesLieInTheSupport(string name)
    {
        var d = Matrix(name);
        var random = new Pcg64DxsmRandom(20261028, 6);
        for (int k = 0; k < 500; k++)
        {
            var x = d.Sample(random);
            Assert.All(x.AsSpan().ToArray(), v => Assert.True(double.IsFinite(v)));
            double logPdf = d.LogPdf(x);

            // ν - p + 1 = 0.2 のウィシャート分布は、数%の標本が double では特異と区別できない（WishartDistribution の説明）
            Assert.True(name == "WishartSmallDegrees" ? !double.IsNaN(logPdf) : double.IsFinite(logPdf), $"sample {k}: {logPdf}");
            if (IsSymmetricValued(d))
            {
                for (int i = 0; i < d.Rows; i++)
                {
                    for (int j = 0; j < i; j++)
                    {
                        Assert.Equal(x[i, j], x[j, i]);
                    }
                }
            }
        }
    }

    [Theory]
    [MemberData(nameof(MatrixNames))]
    public void MatrixMeanIsAFreshCopy(string name)
    {
        var d = Matrix(name);
        var mean = d.GetMean();
        double[] expected = mean.AsSpan().ToArray();
        mean.AsSpan().Fill(42);
        Assert.Equal(expected, d.GetMean().AsSpan().ToArray());
        Assert.Equal(d.Rows, mean.Rows);
        Assert.Equal(d.Columns, mean.Columns);
    }

    // ---------------------------------------------------------------- パラメータの複製

    [Fact]
    public void ParametersAreCopiedAtConstruction()
    {
        double[] mean = (double[])Mean3.Clone();
        double[] covariance = (double[])Covariance3.Clone();
        double[] factor = [1.5, 0.2, -0.3, 0, 0.8, 0.1, 0, 0, 1.2];
        double[] alpha = [2, 3, 0.5];
        double[] matrixMean = [1.0, 0.0, -1.0, 0.5, 2.0, 0.25];
        double[] columnCovariance = [2.0, 0.3, 0.3, 0.5];
        IMultivariateContinuousDistribution[] vectors =
        [
            new MultivariateNormalDistribution(mean, Square(covariance)),
            MultivariateNormalDistribution.FromCholeskyFactor(mean, Square(factor)),
            new MultivariateStudentTDistribution(3.5, mean, Square(covariance)),
            MultivariateStudentTDistribution.FromCholeskyFactor(3.5, mean, Square(factor)),
            new DirichletDistribution(alpha),
        ];
        IMatrixContinuousDistribution[] matrices =
        [
            new WishartDistribution(4.5, Square(covariance)),
            new InverseWishartDistribution(6.5, Square(covariance)),
            new MatrixNormalDistribution(new ReadOnlyMatrixSpan<double>(matrixMean, 3, 2), Square(covariance), Square(columnCovariance)),
        ];
        double[] probabilities = [0.2, 0.3, 0.5];
        double[] failures = [0.25, 0.35];
        IMultivariateDiscreteDistribution[] discretes =
        [
            new MultinomialDistribution(4, probabilities),
            new DirichletMultinomialDistribution(4, alpha),
            new NegativeMultinomialDistribution(2.5, 0.4, failures),
        ];
        double[] point = [0.2, 0.3, 0.5];
        long[] counts = [1, 2, 1];
        var matrixPoints = matrices.Select(m => MatrixPoint(m, 3)).ToArray();
        double[] before =
        [
            .. vectors.Select(v => v.LogPdf(point)), .. matrices.Select((m, i) => m.LogPdf(matrixPoints[i])),
            .. discretes.Select(v => v.LogPmf(counts[..v.Dimension])),
        ];

        foreach (var array in new[] { mean, covariance, factor, alpha, matrixMean, columnCovariance, probabilities, failures })
        {
            array.AsSpan().Fill(double.NaN);
        }

        double[] after =
        [
            .. vectors.Select(v => v.LogPdf(point)), .. matrices.Select((m, i) => m.LogPdf(matrixPoints[i])),
            .. discretes.Select(v => v.LogPmf(counts[..v.Dimension])),
        ];
        Assert.Equal(before, after);
        Assert.All(before, v => Assert.True(double.IsFinite(v)));
    }
}
