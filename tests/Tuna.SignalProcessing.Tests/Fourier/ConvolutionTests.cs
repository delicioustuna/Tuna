using Tuna.LinearAlgebra;
using Tuna.SignalProcessing;
using Complex = System.Numerics.Complex;
using static Tuna.XMath;
using static Tuna.SignalProcessing.XMathExtensions;
using static Tuna.Tests.SignalProcessing.FourierReference;

namespace Tuna.Tests.SignalProcessing;

/// <summary>畳み込み・相関・自己相関（期待値は numpy 2.5.3 の <c>convolve</c>・<c>correlate</c> の出力）</summary>
[Trait("Fourier", "Convolution")]
public class ConvolutionTests
{
    private static readonly ConvolutionMode[] Modes = [ConvolutionMode.Full, ConvolutionMode.Same, ConvolutionMode.Valid];

    private static void AssertClose(ReadOnlyVectorSpan<double> expected, ReadOnlyVectorSpan<double> actual, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance, $"[{i}] expected {expected[i]:R}, actual {actual[i]:R}");
        }
    }

    private static void AssertClose(ReadOnlyVectorSpan<Complex> expected, ReadOnlyVectorSpan<Complex> actual, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(Complex.Abs(expected[i] - actual[i]) <= tolerance, $"[{i}] expected {expected[i]}, actual {actual[i]}");
        }
    }

    [Fact]
    public void RealConvolutionAndCorrelationMatchNumpy()
    {
        var x = new Vector<double>([1, 2, 3, 4, 5]);
        var y = new Vector<double>([2, -1, 0.5]);
        foreach (var method in new[] { ConvolutionMethod.Direct, ConvolutionMethod.Fourier })
        {
            AssertClose(new Vector<double>([2.0, 3, 4.5, 6, 7.5, -3, 2.5]), Convolve(x, y, ConvolutionMode.Full, method), 1e-14);
            AssertClose(new Vector<double>([3.0, 4.5, 6, 7.5, -3]), Convolve(x, y, ConvolutionMode.Same, method), 1e-14);
            AssertClose(new Vector<double>([4.5, 6, 7.5]), Convolve(x, y, ConvolutionMode.Valid, method), 1e-14);
            AssertClose(new Vector<double>([3.0, 4.5, 6, 7.5, -3]), Convolve(y, x, ConvolutionMode.Same, method), 1e-14);

            AssertClose(new Vector<double>([0.5, 0, 1.5, 3, 4.5, 3, 10]), Correlate(x, y, ConvolutionMode.Full, method), 1e-14);
            AssertClose(new Vector<double>([0.0, 1.5, 3, 4.5, 3]), Correlate(x, y, ConvolutionMode.Same, method), 1e-14);
            AssertClose(new Vector<double>([1.5, 3, 4.5]), Correlate(x, y, ConvolutionMode.Valid, method), 1e-14);
            AssertClose(new Vector<double>([10.0, 3, 4.5, 3, 1.5, 0, 0.5]), Correlate(y, x, ConvolutionMode.Full, method), 1e-14);
            AssertClose(new Vector<double>([3.0, 4.5, 3, 1.5, 0]), Correlate(y, x, ConvolutionMode.Same, method), 1e-14);
            AssertClose(new Vector<double>([4.5, 3, 1.5]), Correlate(y, x, ConvolutionMode.Valid, method), 1e-14);

            // x が y より短く、長さが偶数のときの Same（numpy は中央を1つ後ろから取る）
            AssertClose(new Vector<double>([11.0, 8, 5, 2]), Correlate(new Vector<double>([1.0, 2]), new Vector<double>([1.0, 2, 3, 4]), ConvolutionMode.Same, method), 1e-14);
        }
    }

    [Fact]
    public void ComplexCorrelationConjugatesTheSecondInput()
    {
        var a = new Vector<Complex>([new Complex(1, 2), new Complex(0, -1), 3]);
        var b = new Vector<Complex>([new Complex(2, -1), new Complex(1, 1)]);
        foreach (var method in new[] { ConvolutionMethod.Direct, ConvolutionMethod.Fourier })
        {
            AssertClose(new Vector<Complex>([new Complex(3, 1), new Complex(-1, 4), new Complex(4, -5), new Complex(6, 3)]), Correlate(a, b, ConvolutionMode.Full, method), 1e-14);
            AssertClose(new Vector<Complex>([new Complex(3, 1), new Complex(-1, 4), new Complex(4, -5)]), Correlate(a, b, ConvolutionMode.Same, method), 1e-14);
            AssertClose(new Vector<Complex>([new Complex(-1, 4), new Complex(4, -5)]), Correlate(a, b, ConvolutionMode.Valid, method), 1e-14);
            AssertClose(new Vector<Complex>([new Complex(6, -3), new Complex(4, 5), new Complex(-1, -4), new Complex(3, -1)]), Correlate(b, a, ConvolutionMode.Full, method), 1e-14);
            AssertClose(new Vector<Complex>([new Complex(4, 5), new Complex(-1, -4), new Complex(3, -1)]), Correlate(b, a, ConvolutionMode.Same, method), 1e-14);
        }
    }

    [Fact]
    public void IntegerConvolutionIsExact()
    {
        var x = new Vector<int>([1, 2, 3]);
        var y = new Vector<int>([4, 5, 6, 7]);
        Assert.Equal([4, 13, 28, 34, 32, 21], Convolve(x, y).ToArray());
        Assert.Equal([4, 13, 28, 34, 32, 21], Convolve(y, x).ToArray());

        // 多項式の積の係数（大きな整数でも丸めない）
        var p = new Vector<long>([1_000_000_007, 3]);
        Assert.Equal([1_000_000_014_000_000_049L, 24_000_000_168L, 63L], Convolve(p, new Vector<long>([1_000_000_007, 21])).ToArray());
        Assert.Throws<ArgumentException>(() => Convolve(x, y, ConvolutionMode.Full, ConvolutionMethod.Fourier));
    }

    /// <summary>FFT と直接計算の差は ‖x‖₂·‖y‖₂ に対して ε の数十倍以内</summary>
    [Fact]
    public void FourierAndDirectMethodsAgree()
    {
        int[] lengths = [1, 2, 3, 5, 8, 13, 31, 33, 64, 100, 257];
        foreach (int m in lengths)
        {
            foreach (int n in lengths)
            {
                var xr = new Vector<double>(RandomReal(m, m * 1000 + n));
                var yr = new Vector<double>(RandomReal(n, n * 1000 + m));
                var xc = new Vector<Complex>(RandomComplex(m, m * 7 + n));
                var yc = new Vector<Complex>(RandomComplex(n, n * 7 + m));
                double realScale = Norm(xr.AsSpan()) * Norm(yr.AsSpan()) * 64 * Epsilon;
                double complexScale = Norm(xc.AsSpan()) * Norm(yc.AsSpan()) * 64 * Epsilon;
                foreach (var mode in Modes)
                {
                    AssertClose(Convolve(xr, yr, mode, ConvolutionMethod.Direct), Convolve(xr, yr, mode, ConvolutionMethod.Fourier), realScale);
                    AssertClose(Correlate(xr, yr, mode, ConvolutionMethod.Direct), Correlate(xr, yr, mode, ConvolutionMethod.Fourier), realScale);
                    AssertClose(Convolve(xc, yc, mode, ConvolutionMethod.Direct), Convolve(xc, yc, mode, ConvolutionMethod.Fourier), complexScale);
                    AssertClose(Correlate(xc, yc, mode, ConvolutionMethod.Direct), Correlate(xc, yc, mode, ConvolutionMethod.Fourier), complexScale);
                }
            }
        }
    }

    [Fact]
    public void AutoSelectsByLengthOnly()
    {
        var small = new Vector<double>(RandomReal(20, 1));
        var large = new Vector<double>(RandomReal(1000, 2));
        var kernel = new Vector<double>(RandomReal(200, 3));
        Assert.Equal(Convolve(small, kernel, ConvolutionMode.Full, ConvolutionMethod.Direct).ToArray(), Convolve(small, kernel).ToArray());
        Assert.Equal(Convolve(large, kernel, ConvolutionMode.Full, ConvolutionMethod.Fourier).ToArray(), Convolve(large, kernel).ToArray());
        Assert.Equal(AutoCorrelate(large, 10, ConvolutionMethod.Direct).ToArray(), AutoCorrelate(large, 10).ToArray());
        Assert.Equal(AutoCorrelate(large, 500, ConvolutionMethod.Fourier).ToArray(), AutoCorrelate(large, 500).ToArray());
    }

    [Fact]
    public void AutoCorrelationMatchesCorrelationAndDirectSums()
    {
        foreach (int n in new[] { 1, 2, 7, 64, 100, 1000 })
        {
            var x = new Vector<double>(RandomReal(n, n));
            var full = Correlate(x, x, ConvolutionMode.Full, ConvolutionMethod.Direct);
            var c = new Vector<Complex>(RandomComplex(n, n + 1));
            var complexFull = Correlate(c, c, ConvolutionMode.Full, ConvolutionMethod.Direct);
            double scale = Norm(x.AsSpan()) * Norm(x.AsSpan()) * 64 * Epsilon;
            double complexScale = Norm(c.AsSpan()) * Norm(c.AsSpan()) * 64 * Epsilon;
            foreach (int maxLag in new[] { 0, n / 2, n - 1 })
            {
                var direct = AutoCorrelate(x, maxLag, ConvolutionMethod.Direct);
                Assert.Equal(full.AsSpan().Slice(n - 1, maxLag + 1).ToArray(), direct.ToArray());
                AssertClose(direct, AutoCorrelate(x, maxLag, ConvolutionMethod.Fourier), scale);

                var complexDirect = AutoCorrelate(c, maxLag, ConvolutionMethod.Direct);
                Assert.Equal(complexFull.AsSpan().Slice(n - 1, maxLag + 1).ToArray(), complexDirect.ToArray());
                AssertClose(complexDirect, AutoCorrelate(c, maxLag, ConvolutionMethod.Fourier), complexScale);
            }
        }
    }

    [Fact]
    public void ColumnAndRowAutoCorrelationsMatchVectorAutoCorrelationsBitwise()
    {
        var draws = Matrix<double>.FromColumnMajor(500, 4, RandomReal(2000, 5));
        foreach (var method in new[] { ConvolutionMethod.Auto, ConvolutionMethod.Direct, ConvolutionMethod.Fourier })
        {
            var columns = ColumnAutoCorrelations(draws, 50, method);
            Assert.Equal(51, columns.Rows);
            Assert.Equal(4, columns.Columns);
            var rows = RowAutoCorrelations(draws.T, 50, method);
            Assert.Equal(4, rows.Rows);
            Assert.Equal(51, rows.Columns);
            for (int j = 0; j < 4; j++)
            {
                var expected = AutoCorrelate(draws.Column(j).ToVector(), 50, method).ToArray();
                Assert.Equal(expected, columns.Column(j).ToArray());
                Assert.Equal(expected, rows.Row(j).ToArray());
            }
        }
    }

    [Fact]
    public void StridedInputsAndDestinations()
    {
        var m = Matrix<double>.FromColumnMajor(6, 40, RandomReal(240, 9));
        var x = m.Row(1);
        var y = m.Row(4)[..35];
        var expected = Convolve(x.ToVector(), y.ToVector(), ConvolutionMode.Full, ConvolutionMethod.Fourier);
        var target = new Matrix<double>(3, 74);
        Convolve(x, y, ConvolutionMode.Full, ConvolutionMethod.Fourier, target.Row(2));
        Assert.Equal(expected.ToArray(), target.Row(2).ToArray());
        var direct = Convolve(x.ToVector(), y.ToVector(), ConvolutionMode.Same, ConvolutionMethod.Direct);
        Assert.Equal(direct.ToArray(), Convolve(x, y, ConvolutionMode.Same, ConvolutionMethod.Direct).ToArray());
    }

    [Fact]
    public void InvalidArgumentsThrow()
    {
        var x = new Vector<double>([1.0, 2, 3]);
        Assert.Throws<ArgumentException>(() => Convolve(x, new Vector<double>(0)));
        Assert.Throws<ArgumentException>(() => Convolve(x, x, new Vector<double>(4)));
        Assert.Throws<ArgumentException>(() => Convolve(x, x, x));
        Assert.Throws<ArgumentOutOfRangeException>(() => AutoCorrelate(x, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => AutoCorrelate(x, -1));
        Assert.Throws<ArgumentException>(() => AutoCorrelate(new Vector<double>(0), 0));
        Assert.Throws<ArgumentException>(() => AutoCorrelate(new Vector<int>([1, 2]), 1, ConvolutionMethod.Fourier));
        Assert.Throws<ArgumentOutOfRangeException>(() => Convolve(x, x, (ConvolutionMode)5));
        Assert.Throws<ArgumentOutOfRangeException>(() => Convolve(x, x, ConvolutionMode.Full, (ConvolutionMethod)5));
    }

    private static double Norm(ReadOnlySpan<double> x)
    {
        double sum = 0;
        foreach (double v in x)
        {
            sum += v * v;
        }

        return Math.Sqrt(sum);
    }

    private static double Norm(ReadOnlySpan<Complex> x)
    {
        double sum = 0;
        foreach (var v in x)
        {
            sum += v.Real * v.Real + v.Imaginary * v.Imaginary;
        }

        return Math.Sqrt(sum);
    }
}
