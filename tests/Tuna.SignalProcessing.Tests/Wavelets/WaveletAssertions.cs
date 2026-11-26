using Tuna.LinearAlgebra;

namespace Tuna.Tests.SignalProcessing;

/// <summary>ウェーブレットの拡張のテストに共通の比較</summary>
internal static class WaveletAssertions
{
    /// <summary>
    /// 照合の許容誤差（期待値の最大の大きさに対する比）。PyWavelets の Symlet と bior4.4・5.5・6.8 の表は定義式の値と最大 1.5e−11 違うので緩める
    /// </summary>
    internal static double Tolerance(string name, double scale)
    {
        bool tabulated = name.StartsWith("sym", StringComparison.Ordinal) || name.EndsWith("4.4", StringComparison.Ordinal)
            || name.EndsWith("5.5", StringComparison.Ordinal) || name.EndsWith("6.8", StringComparison.Ordinal);
        return (tabulated ? 1e-10 : 1e-12) * Math.Max(scale, 1);
    }

    internal static double MaxAbs(ReadOnlySpan<double> values)
    {
        double scale = 0;
        foreach (double v in values)
        {
            scale = Math.Max(scale, Math.Abs(v));
        }

        return scale;
    }

    internal static double MaxAbs(double[,] values)
    {
        double scale = 0;
        foreach (double v in values)
        {
            scale = Math.Max(scale, Math.Abs(v));
        }

        return scale;
    }

    internal static void Close(ReadOnlySpan<double> expected, ReadOnlyVectorSpan<double> actual, double tolerance, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (!(Math.Abs(expected[i] - actual[i]) <= tolerance))
            {
                Assert.Fail($"{label}[{i}]: expected {expected[i]:R}, actual {actual[i]:R} (tolerance {tolerance:E1})");
            }
        }
    }

    internal static void Close(double[,] expected, ReadOnlyMatrixSpan<double> actual, double tolerance, string label)
    {
        Assert.Equal(expected.GetLength(0), actual.Rows);
        Assert.Equal(expected.GetLength(1), actual.Columns);
        for (int i = 0; i < actual.Rows; i++)
        {
            for (int j = 0; j < actual.Columns; j++)
            {
                if (!(Math.Abs(expected[i, j] - actual[i, j]) <= tolerance))
                {
                    Assert.Fail($"{label}[{i}, {j}]: expected {expected[i, j]:R}, actual {actual[i, j]:R} (tolerance {tolerance:E1})");
                }
            }
        }
    }

    /// <summary>ビット単位で同じ（NaN も同じビット列）</summary>
    internal static void Identical(ReadOnlyMatrixSpan<double> expected, ReadOnlyMatrixSpan<double> actual, string label)
    {
        Assert.Equal(expected.Rows, actual.Rows);
        Assert.Equal(expected.Columns, actual.Columns);
        for (int i = 0; i < actual.Rows; i++)
        {
            for (int j = 0; j < actual.Columns; j++)
            {
                if (BitConverter.DoubleToInt64Bits(expected[i, j]) != BitConverter.DoubleToInt64Bits(actual[i, j]))
                {
                    Assert.Fail($"{label}[{i}, {j}]: expected {expected[i, j]:R}, actual {actual[i, j]:R}");
                }
            }
        }
    }

    internal static void Identical(ReadOnlyVectorSpan<double> expected, ReadOnlyVectorSpan<double> actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < actual.Length; i++)
        {
            if (BitConverter.DoubleToInt64Bits(expected[i]) != BitConverter.DoubleToInt64Bits(actual[i]))
            {
                Assert.Fail($"{label}[{i}]: expected {expected[i]:R}, actual {actual[i]:R}");
            }
        }
    }

    internal static double[] RandomSignal(int length, int seed)
    {
        var random = new Random(seed);
        var x = new double[length];
        for (int i = 0; i < length; i++)
        {
            x[i] = 2 * random.NextDouble() - 1;
        }

        return x;
    }

    internal static Matrix<double> RandomMatrix(int rows, int columns, int seed)
    {
        var random = new Random(seed);
        var x = new Matrix<double>(rows, columns);
        for (int j = 0; j < columns; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                x[i, j] = 2 * random.NextDouble() - 1;
            }
        }

        return x;
    }

    internal static double[] ToArray(ReadOnlyVectorSpan<double> x)
    {
        var result = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            result[i] = x[i];
        }

        return result;
    }
}
