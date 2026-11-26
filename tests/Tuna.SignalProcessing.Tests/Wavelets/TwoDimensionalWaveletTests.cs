using Tuna.LinearAlgebra;
using Tuna.SignalProcessing;
using static Tuna.Tests.SignalProcessing.WaveletAssertions;

namespace Tuna.Tests.SignalProcessing;

/// <summary>各行・各列の離散ウェーブレット変換と2次元の変換</summary>
[Trait("Wavelets", "TwoDimensional")]
public class TwoDimensionalWaveletTests
{
    private static readonly WaveletBoundary[] Boundaries =
    [
        WaveletBoundary.Symmetric, WaveletBoundary.Periodization, WaveletBoundary.Zero,
        WaveletBoundary.Constant, WaveletBoundary.Reflect, WaveletBoundary.Periodic,
    ];

    [Fact]
    public void AxisTransformsMatchPyWavelets()
    {
        foreach (var (name, boundary, axis, x, expectedA, expectedD, expectedInverse) in WaveletExtensionReferences.Axis)
        {
            var w = DiscreteWavelet.FromName(name);
            var matrix = new Matrix<double>(x);
            var (a, d) = axis == 1 ? Wavelets.ForwardRows(matrix, w, boundary) : Wavelets.ForwardColumns(matrix, w, boundary);
            string label = $"{name} {boundary} axis {axis}";
            Close(expectedA, a, Tolerance(name, MaxAbs(expectedA)), label + " a");
            Close(expectedD, d, Tolerance(name, MaxAbs(expectedD)), label + " d");

            var scaledA = 1.5 * a;
            var scaledD = 0.5 * d;
            var inverse = axis == 1 ? Wavelets.InverseRows(scaledA, scaledD, w, boundary) : Wavelets.InverseColumns(scaledA, scaledD, w, boundary);
            Close(expectedInverse, inverse, Tolerance(name, MaxAbs(expectedInverse)), label + " inverse");
        }
    }

    [Fact]
    public void Forward2DMatchesPyWavelets()
    {
        foreach (var (name, boundary, x, expectedA, expectedH, expectedV, expectedD) in WaveletExtensionReferences.Forward2D)
        {
            var w = DiscreteWavelet.FromName(name);
            var (a, h, v, d) = Wavelets.Forward2D(new Matrix<double>(x), w, boundary);
            string label = $"{name} {boundary}";
            Close(expectedA, a, Tolerance(name, MaxAbs(expectedA)), label + " cA");
            Close(expectedH, h, Tolerance(name, MaxAbs(expectedH)), label + " cH");
            Close(expectedV, v, Tolerance(name, MaxAbs(expectedV)), label + " cV");
            Close(expectedD, d, Tolerance(name, MaxAbs(expectedD)), label + " cD");
        }
    }

    [Fact]
    public void Inverse2DMatchesPyWavelets()
    {
        foreach (var (name, boundary, a, h, v, d, expected) in WaveletExtensionReferences.Inverse2D)
        {
            var w = DiscreteWavelet.FromName(name);
            var result = Wavelets.Inverse2D(new Matrix<double>(a), new Matrix<double>(h), new Matrix<double>(v), new Matrix<double>(d), w, boundary);
            Close(expected, result, Tolerance(name, MaxAbs(expected)), $"{name} {boundary}");
        }
    }

    [Fact]
    public void Decompose2DMatchesPyWavelets()
    {
        foreach (var (name, boundary, levels, x, expectedA, details) in WaveletExtensionReferences.Decompose2D)
        {
            var w = DiscreteWavelet.FromName(name);
            var matrix = new Matrix<double>(x);
            var decomposition = levels < 0 ? Wavelets.Decompose2D(matrix, w, boundary) : Wavelets.Decompose2D(matrix, w, levels, boundary);
            string label = $"{name} {boundary}";
            Assert.Equal(details.Length, decomposition.Levels);
            Assert.Equal(x.GetLength(0), decomposition.Rows);
            Assert.Equal(x.GetLength(1), decomposition.Columns);
            Close(expectedA, decomposition.Approximation, Tolerance(name, MaxAbs(expectedA)), label + " cA");
            for (int i = 0; i < details.Length; i++)
            {
                var (h, v, d) = decomposition.Details[i];
                Close(details[i].H, h, Tolerance(name, MaxAbs(details[i].H)), $"{label} cH[{i}]");
                Close(details[i].V, v, Tolerance(name, MaxAbs(details[i].V)), $"{label} cV[{i}]");
                Close(details[i].D, d, Tolerance(name, MaxAbs(details[i].D)), $"{label} cD[{i}]");
            }

            Assert.Same(decomposition.Details[0].Diagonal, decomposition.Detail(decomposition.Levels).Diagonal);
            Close(x, Wavelets.Reconstruct(decomposition), 1e-12, label + " reconstruct");
        }
    }

    [Theory]
    [InlineData("db2")]
    [InlineData("bior2.4")]
    [InlineData("sym5")]
    public void Decompose2DRoundTripsForOddShapes(string name)
    {
        var w = DiscreteWavelet.FromName(name);
        foreach (var boundary in Boundaries)
        {
            foreach (var (rows, columns) in new[] { (1, 9), (17, 23), (40, 3) })
            {
                if (boundary == WaveletBoundary.Reflect && Math.Min(rows, columns) < 2)
                {
                    continue;
                }

                var x = RandomMatrix(rows, columns, rows * 100 + columns);
                int levels = boundary == WaveletBoundary.Reflect ? 1 : 3;
                var decomposition = Wavelets.Decompose2D(x, w, levels, boundary);
                Close(x.ToArray(), Wavelets.Reconstruct(decomposition), 1e-12, $"{name} {boundary} {rows}x{columns}");
            }
        }

        Assert.Equal(Math.Min(Wavelets.MaxLevel(100, w), Wavelets.MaxLevel(30, w)), Wavelets.MaxLevel(100, 30, w));
    }

    [Theory]
    [InlineData(3, 29)]
    [InlineData(37, 16)]
    [InlineData(1100, 11)]
    public void RowAndColumnTransformsEqualTheOneDimensionalTransformBitwise(int rows, int columns)
    {
        // 行の向きの SIMD（多い行）と1行ずつの変換（少ない行）のどちらでも、1次元の変換とビット単位で同じ
        var x = RandomMatrix(rows, columns, rows + columns);
        foreach (string name in new[] { "haar", "db3", "bior3.5" })
        {
            var w = DiscreteWavelet.FromName(name);
            foreach (var boundary in Boundaries)
            {
                string label = $"{name} {boundary} {rows}x{columns}";
                var (a, d) = Wavelets.ForwardRows(x, w, boundary);
                var back = Wavelets.InverseRows(a, d, w, boundary);
                for (int i = 0; i < rows; i++)
                {
                    var (ai, di) = Wavelets.Forward(x.Row(i), w, boundary);
                    Identical(ai, a.Row(i), $"{label} row {i} a");
                    Identical(di, d.Row(i), $"{label} row {i} d");
                    Identical(Wavelets.Inverse(a.Row(i), d.Row(i), w, boundary), back.Row(i), $"{label} row {i} inverse");
                }

                var (ac, dc) = Wavelets.ForwardColumns(x.T, w, boundary);
                var backColumns = Wavelets.InverseColumns(ac, dc, w, boundary);
                Identical(a.T, ac, label + " columns a");
                Identical(d.T, dc, label + " columns d");
                Identical(back.T, backColumns, label + " columns inverse");
            }
        }
    }

    [Fact]
    public void ViewsGiveTheSameResultAsDenseCopies()
    {
        var big = RandomMatrix(40, 50, 12);
        var view = big[3..38, 5..47];
        var dense = view.ToMatrix();
        var w = DiscreteWavelet.Daubechies(4);
        var (a1, h1, v1, d1) = Wavelets.Forward2D(view, w);
        var (a2, h2, v2, d2) = Wavelets.Forward2D(dense, w);
        Identical(a2, a1, "a");
        Identical(h2, h1, "h");
        Identical(v2, v1, "v");
        Identical(d2, d1, "d");

        // 転置のビュー（行が連続）
        var (ar, dr) = Wavelets.ForwardRows(dense.T, w);
        var (ac, dc) = Wavelets.ForwardColumns(dense, w);
        Identical(ac.T, ar, "transposed a");
        Identical(dc.T, dr, "transposed d");
    }

    [Fact]
    public void Forward2DIsSeparable()
    {
        // 2次元の変換は列の変換の後の行の変換。近似と垂直は列の低域、水平と対角は列の高域
        var x = RandomMatrix(12, 15, 3);
        var w = DiscreteWavelet.Symlet(3);
        var (a, h, v, d) = Wavelets.Forward2D(x, w);
        var (low, high) = Wavelets.ForwardColumns(x, w);
        var (lowLow, lowHigh) = Wavelets.ForwardRows(low, w);
        var (highLow, highHigh) = Wavelets.ForwardRows(high, w);
        Identical(lowLow, a, "approximation");
        Identical(highLow, h, "horizontal");
        Identical(lowHigh, v, "vertical");
        Identical(highHigh, d, "diagonal");
        Close(x.ToArray(), Wavelets.Inverse2D(a, h, v, d, w)[..12, ..15], 1e-13, "inverse");
    }

    [Fact]
    public void TwoDimensionalTransformsValidateArguments()
    {
        var w = DiscreteWavelet.Daubechies(2);
        Assert.Throws<ArgumentException>(() => Wavelets.ForwardRows(new Matrix<double>(0, 4), w));
        Assert.Throws<ArgumentException>(() => Wavelets.ForwardRows(new Matrix<double>(4, 1), w, WaveletBoundary.Reflect));
        Assert.Throws<ArgumentException>(() => Wavelets.ForwardColumns(new Matrix<double>(1, 4), w, WaveletBoundary.Reflect));
        Assert.Throws<ArgumentException>(() => Wavelets.InverseRows(new Matrix<double>(3, 4), new Matrix<double>(3, 5), w));
        Assert.Throws<ArgumentException>(() => Wavelets.Inverse2D(
            new Matrix<double>(3, 4), new Matrix<double>(3, 4), new Matrix<double>(3, 4), new Matrix<double>(4, 4), w));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.Decompose2D(new Matrix<double>(4, 4), w, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.Decompose2D(new Matrix<double>(2, 8), DiscreteWavelet.Haar, 2, WaveletBoundary.Reflect));
        Assert.Throws<ArgumentNullException>(() => Wavelets.Reconstruct((WaveletDecomposition2D)null!));

        var zero = Wavelets.Decompose2D(RandomMatrix(5, 6, 1), w, 0);
        Assert.Equal(0, zero.Levels);
        Assert.Throws<ArgumentOutOfRangeException>(() => zero.Detail(1));
        Close(RandomMatrix(5, 6, 1).ToArray(), Wavelets.Reconstruct(zero), 0, "levels 0");
    }
}
