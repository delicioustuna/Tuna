using System.Numerics;
using Tuna.LinearAlgebra;

namespace Tuna.NeuralNetwork.Tests;

/// <summary>乱数の入力と、結果の比べ方</summary>
internal static class TestData
{
    /// <summary>[-scale, scale) の一様乱数（種で決まる）</summary>
    public static double[] Uniform(int length, int seed, double scale = 1)
    {
        var random = new Random(seed);
        var values = new double[length];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = scale * (2 * random.NextDouble() - 1);
        }

        return values;
    }

    /// <summary>[low, high) の一様乱数</summary>
    public static double[] Uniform(int length, int seed, double low, double high)
    {
        var random = new Random(seed);
        var values = new double[length];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = low + (high - low) * random.NextDouble();
        }

        return values;
    }

    /// <summary>要素型を変える（float にするなら丸めた値を基準にも使う）</summary>
    public static T[] To<T>(double[] values) where T : INumberBase<T> => values.Select(T.CreateTruncating).ToArray();

    /// <summary>要素型 T に丸めた値を double に戻す（基準の計算を同じ入力で行うため）</summary>
    public static double[] Rounded<T>(double[] values) where T : INumberBase<T> =>
        values.Select(v => double.CreateTruncating(T.CreateTruncating(v))).ToArray();

    public static LinearAlgebra.Vector<T> Vector<T>(double[] values) where T : unmanaged, INumberBase<T> => new(To<T>(values));

    public static Matrix<T> Matrix<T>(double[] columnMajor, int rows, int columns) where T : unmanaged, INumberBase<T> =>
        LinearAlgebra.Matrix<T>.FromColumnMajor(rows, columns, To<T>(columnMajor));

    /// <summary>要素型ごとの許容誤差（最大値で正規化した誤差）</summary>
    public static double Tolerance<T>() => typeof(T) == typeof(float) ? 2e-5 : 1e-12;

    /// <summary>|actual − expected| ≤ tolerance·max(1, max|expected|) をすべての要素で</summary>
    public static void Close<T>(double[] expected, ReadOnlyMatrixSpan<T> actual, double tolerance) where T : unmanaged, INumberBase<T>
    {
        Assert.Equal(expected.Length, actual.Rows * actual.Columns);
        double scale = Math.Max(1, expected.Select(Math.Abs).DefaultIfEmpty().Max());
        double worst = 0;
        for (int j = 0; j < actual.Columns; j++)
        {
            for (int i = 0; i < actual.Rows; i++)
            {
                double e = expected[j * actual.Rows + i], a = double.CreateTruncating(actual[i, j]);
                Assert.False(double.IsNaN(a), $"NaN at ({i}, {j})");
                worst = Math.Max(worst, Math.Abs(a - e));
            }
        }

        Assert.True(worst <= tolerance * scale, $"max error {worst:E3} > {tolerance * scale:E3}");
    }

    /// <summary>テンソル [rows, columns, count] を列優先の配列と比べる</summary>
    public static void Close<T>(double[] expected, ReadOnlyTensorSpan<T> actual, double tolerance) where T : unmanaged, INumberBase<T>
    {
        int rows = (int)actual.Lengths[0], columns = (int)actual.Lengths[1];
        Assert.Equal(expected.Length, rows * columns * actual.Matrices.Count);
        for (int k = 0; k < actual.Matrices.Count; k++)
        {
            Close(expected.AsSpan(k * rows * columns, rows * columns).ToArray(), actual.GetMatrix(k), tolerance);
        }
    }

    /// <summary>ビット単位で同じ</summary>
    public static void Same<T>(ReadOnlyMatrixSpan<T> expected, ReadOnlyMatrixSpan<T> actual) where T : unmanaged, INumberBase<T>
    {
        Assert.Equal(expected.Rows, actual.Rows);
        Assert.Equal(expected.Columns, actual.Columns);
        for (int j = 0; j < expected.Columns; j++)
        {
            for (int i = 0; i < expected.Rows; i++)
            {
                T e = expected[i, j], a = actual[i, j];
                Assert.True(Bits(e) == Bits(a), $"({i}, {j}): {e} != {a}");
            }
        }
    }

    public static void Same<T>(ReadOnlyTensorSpan<T> expected, ReadOnlyTensorSpan<T> actual) where T : unmanaged, INumberBase<T>
    {
        Assert.Equal(expected.Matrices.Count, actual.Matrices.Count);
        for (int k = 0; k < expected.Matrices.Count; k++)
        {
            Same(expected.GetMatrix(k), actual.GetMatrix(k));
        }
    }

    private static long Bits<T>(T value) where T : unmanaged =>
        typeof(T) == typeof(float) ? BitConverter.SingleToInt32Bits((float)(object)value) : BitConverter.DoubleToInt64Bits((double)(object)value);

    /// <summary>並列の数を変えて実行する</summary>
    public static TResult WithThreads<TResult>(int threads, Func<TResult> action)
    {
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = threads;
            return action();
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }
}
