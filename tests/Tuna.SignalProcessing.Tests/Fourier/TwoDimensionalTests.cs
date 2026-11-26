using Tuna.LinearAlgebra;
using Tuna.SignalProcessing;
using Complex = System.Numerics.Complex;
using static Tuna.Tests.SignalProcessing.FourierReference;

namespace Tuna.Tests.SignalProcessing;

/// <summary>2次元の FFT と中央への並べ替え: numpy との照合、行・列の変換とのビット一致、逆変換、倍率、例外</summary>
[Trait("Fourier", "TwoDimensional")]
public class TwoDimensionalTests
{
    private static readonly (int Rows, int Columns)[] Shapes = [(1, 1), (1, 8), (8, 1), (2, 3), (5, 7), (8, 6), (16, 31), (37, 12)];

    private static Matrix<Complex> RandomMatrix(int rows, int columns, int seed) =>
        Matrix<Complex>.FromColumnMajor(rows, columns, RandomComplex(rows * columns, seed));

    private static Matrix<double> RandomRealMatrix(int rows, int columns, int seed) =>
        Matrix<double>.FromColumnMajor(rows, columns, RandomReal(rows * columns, seed));

    private static void AssertBitwiseEqual(ReadOnlyMatrixSpan<Complex> expected, ReadOnlyMatrixSpan<Complex> actual, string label)
    {
        Assert.Equal(expected.Rows, actual.Rows);
        Assert.Equal(expected.Columns, actual.Columns);
        for (int i = 0; i < expected.Rows; i++)
        {
            for (int j = 0; j < expected.Columns; j++)
            {
                if (BitConverter.DoubleToInt64Bits(expected[i, j].Real) != BitConverter.DoubleToInt64Bits(actual[i, j].Real)
                    || BitConverter.DoubleToInt64Bits(expected[i, j].Imaginary) != BitConverter.DoubleToInt64Bits(actual[i, j].Imaginary))
                {
                    Assert.Fail($"{label} [{i},{j}] expected {expected[i, j]}, actual {actual[i, j]}");
                }
            }
        }
    }

    private static void AssertClose(ReadOnlySpan<Complex> expected, ReadOnlyMatrixSpan<Complex> actual, double tolerance, string label)
    {
        Assert.Equal(expected.Length, actual.Rows * actual.Columns);
        for (int j = 0; j < actual.Columns; j++)
        {
            for (int i = 0; i < actual.Rows; i++)
            {
                var e = expected[i + actual.Rows * j];
                Assert.True((actual[i, j] - e).Magnitude <= tolerance * (1 + e.Magnitude), $"{label} [{i},{j}]: {actual[i, j]} vs {e}");
            }
        }
    }

    private static void AssertClose(ReadOnlySpan<double> expected, ReadOnlyMatrixSpan<double> actual, double tolerance, string label)
    {
        Assert.Equal(expected.Length, actual.Rows * actual.Columns);
        for (int j = 0; j < actual.Columns; j++)
        {
            for (int i = 0; i < actual.Rows; i++)
            {
                double e = expected[i + actual.Rows * j];
                Assert.True(Math.Abs(actual[i, j] - e) <= tolerance * (1 + Math.Abs(e)), $"{label} [{i},{j}]: {actual[i, j]:R} vs {e:R}");
            }
        }
    }

    [Fact]
    public void MatchesNumpy()
    {
        var x = Matrix<Complex>.FromColumnMajor(3, 5, SciPyReferences.ComplexMatrix);
        AssertClose(SciPyReferences.Fft2, Fourier.Forward2D(x), 1e-14, "fft2");
        AssertClose(SciPyReferences.Ifft2, Fourier.Inverse2D(x), 1e-14, "ifft2");

        var real = Matrix<double>.FromColumnMajor(4, 6, SciPyReferences.RealMatrix);
        var spectrum = Fourier.ForwardReal2D(real);
        Assert.Equal(4, spectrum.Rows);
        Assert.Equal(4, spectrum.Columns);
        AssertClose(SciPyReferences.Rfft2, spectrum, 1e-14, "rfft2");

        // 片側スペクトルに複素数を掛けて第0列にも虚部を持たせ、numpy と同じ順（列方向の逆変換が先）に捨てることを確かめる
        var odd = Matrix<double>.FromColumnMajor(5, 7, SciPyReferences.OddRealMatrix);
        var oddSpectrum = Fourier.ForwardReal2D(odd);
        AssertClose(SciPyReferences.OddRfft2, oddSpectrum, 1e-14, "rfft2 odd");
        var modified = Matrix<Complex>.FromColumnMajor(5, 4, SciPyReferences.OddRfft2.Select(v => v * new Complex(1, 0.5)).ToArray());
        AssertClose(SciPyReferences.OddIrfft2, Fourier.InverseReal2D(modified, 7), 1e-14, "irfft2");
    }

    [Fact]
    public void MatchesRowAndColumnTransformsBitwise()
    {
        foreach (var (rows, columns) in Shapes)
        {
            string label = $"{rows}x{columns}";
            var x = RandomMatrix(rows, columns, rows * 100 + columns);
            foreach (var normalization in new[] { FourierNormalization.Backward, FourierNormalization.Orthonormal, FourierNormalization.Forward })
            {
                var forward = Fourier.Forward2D(x, normalization);
                AssertBitwiseEqual(Fourier.ForwardColumns(Fourier.ForwardRows(x, normalization), normalization), forward, $"{label} forward");
                var inverse = Fourier.Inverse2D(x, normalization);
                AssertBitwiseEqual(Fourier.InverseRows(Fourier.InverseColumns(x, normalization), normalization), inverse, $"{label} inverse");

                // その場更新と、転置ビュー（行が連続でない配置）
                var copy = x.AsMatrixSpan().ToMatrix();
                Fourier.InPlace.Forward2D(copy, normalization);
                AssertBitwiseEqual(forward, copy, $"{label} in place");
                Fourier.InPlace.Inverse2D(copy, normalization);
                var transposed = Fourier.Forward2D(x.T, normalization);
                AssertBitwiseEqual(Fourier.ForwardColumns(Fourier.ForwardRows(x.T, normalization), normalization), transposed, $"{label} transposed");

                var real = RandomRealMatrix(rows, columns, rows * 7 + columns);
                var spectrum = Fourier.ForwardReal2D(real, normalization);
                AssertBitwiseEqual(Fourier.ForwardColumns(Fourier.ForwardRealRows(real, normalization), normalization), spectrum, $"{label} real");
            }
        }
    }

    [Fact]
    public void InverseRestoresInput()
    {
        foreach (var (rows, columns) in Shapes)
        {
            var x = RandomMatrix(rows, columns, rows + 3 * columns);
            var back = Fourier.Inverse2D(Fourier.Forward2D(x));
            var real = RandomRealMatrix(rows, columns, rows + 5 * columns);
            var realBack = Fourier.InverseReal2D(Fourier.ForwardReal2D(real), columns);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    Assert.True((back[i, j] - x[i, j]).Magnitude < 1e-14, $"{rows}x{columns} [{i},{j}]");
                    Assert.True(Math.Abs(realBack[i, j] - real[i, j]) < 1e-14, $"{rows}x{columns} real [{i},{j}]");
                }
            }
        }
    }

    [Fact]
    public void OrthonormalTransformPreservesEnergy()
    {
        var x = RandomMatrix(9, 14, 3);
        var y = Fourier.Forward2D(x, FourierNormalization.Orthonormal);
        double energy = 0, transformed = 0;
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 14; j++)
            {
                energy += x[i, j].Magnitude * x[i, j].Magnitude;
                transformed += y[i, j].Magnitude * y[i, j].Magnitude;
            }
        }

        Assert.Equal(energy, transformed, 1e-12 * energy);
    }

    [Fact]
    public void ShiftMovesTheZeroFrequencyToTheCenter()
    {
        foreach (var (rows, columns) in Shapes)
        {
            var x = RandomRealMatrix(rows, columns, rows * 13 + columns);
            var shifted = Fourier.Shift2D(x);
            var expected = Fourier.ShiftRows(Fourier.ShiftColumns(x));
            var back = Fourier.InverseShift2D(shifted);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    Assert.Equal(expected[i, j], shifted[i, j]);
                    Assert.Equal(x[i, j], back[i, j]);
                }
            }

            // 周波数 0 は中央（⌊R/2⌋, ⌊C/2⌋）へ
            Assert.Equal(x[0, 0], shifted[rows / 2, columns / 2]);
        }
    }

    [Fact]
    public void RejectsInvalidShapesAndOverlap()
    {
        var x = RandomMatrix(4, 6, 1);
        Assert.Throws<ArgumentException>(() => Fourier.Forward2D(x, new Matrix<Complex>(4, 5)));
        Assert.Throws<ArgumentException>(() => Fourier.ForwardReal2D(RandomRealMatrix(4, 6, 2), new Matrix<Complex>(4, 3)));
        Assert.Throws<ArgumentException>(() => Fourier.InverseReal2D(new Matrix<Complex>(4, 4), new Matrix<double>(4, 5)));
        Assert.Throws<ArgumentException>(() => Fourier.InverseReal2D(new Matrix<Complex>(4, 4), new Matrix<double>(3, 6)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fourier.InverseReal2D(new Matrix<Complex>(4, 4), -1));
        Assert.Throws<ArgumentException>(() => Fourier.Shift2D(x, x));
        Assert.Throws<ArgumentException>(() => Fourier.Shift2D(x, new Matrix<Complex>(6, 4)));

        // 空の行列
        Assert.Equal(0, Fourier.Forward2D(new Matrix<Complex>(0, 3)).Rows);
        Assert.Equal(0, Fourier.ForwardReal2D(new Matrix<double>(3, 0)).Columns);
        Assert.Equal(0, Fourier.Shift2D(new Matrix<double>(0, 0)).Rows);
    }
}
