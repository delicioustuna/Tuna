using Tuna.LinearAlgebra;
using Tuna.SignalProcessing;
using Complex = System.Numerics.Complex;
using static Tuna.Tests.SignalProcessing.FourierReference;

namespace Tuna.Tests.SignalProcessing;

/// <summary>
/// 作業スレッドに分けた経路（Stockham の各段、実数 FFT の前後処理、行列の行ごとの変換、DCT の行、CWT のスケール）が、
/// 1スレッドとビット単位で同じ結果を返すことを確かめる。<see cref="LinearAlgebraSettings.MaxDegreeOfParallelism"/> を書き換えるので、他のテストと同時に走らせない
/// </summary>
[Trait("Fourier", "Parallelism")]
[Collection(nameof(ParallelismCollection))]
public class FourierParallelismTests
{
    /// <summary>
    /// 並列化される大きさで、基数 8 の第1段の端数（8·3⁹ は m が奇数）、基数 3・5・一般の奇数基数（7⁶）、Bluestein 法（65537）を通る長さ
    /// </summary>
    public static TheoryData<int> Lengths => new() { 131072, 157464, 177147, 117649, 156250, 65537 };

    [Theory]
    [MemberData(nameof(Lengths))]
    public void VectorTransformsMatchSingleThreadBitwise(int n)
    {
        var x = RandomComplex(n, n);
        var real = RandomReal(n, n + 1);
        var spectrum = Fourier.ForwardReal(new ReadOnlyVectorSpan<double>(real));
        Compare(() => Fourier.Forward(new ReadOnlyVectorSpan<Complex>(x)).ToArray(), "forward");
        Compare(() => Fourier.Inverse(new ReadOnlyVectorSpan<Complex>(x)).ToArray(), "inverse");
        Compare(() => Fourier.ForwardReal(new ReadOnlyVectorSpan<double>(real)).ToArray(), "forward real");
        Compare(() => Fourier.InverseReal(spectrum, n).ToArray(), "inverse real");
    }

    [Fact]
    public void MatrixTransformsMatchSingleThreadBitwise()
    {
        var dense = Matrix<Complex>.FromColumnMajor(64, 2048, RandomComplex(64 * 2048, 1));
        var tall = Matrix<Complex>.FromColumnMajor(2048, 64, RandomComplex(2048 * 64, 2));
        var padded = Matrix<Complex>.FromColumnMajor(65, 2048, RandomComplex(65 * 2048, 3));
        var real = Matrix<double>.FromColumnMajor(64, 2050, RandomReal(64 * 2050, 4));
        var realPadded = Matrix<double>.FromColumnMajor(65, 2050, RandomReal(65 * 2050, 5));
        var square = Matrix<double>.FromColumnMajor(512, 512, RandomReal(512 * 512, 6));

        // 列優先で連続（そのまま一括変換）、各列が連続（1行ずつ）、ストライド付きのビュー（集めて変換）
        Compare(() => Fourier.ForwardRows(dense).AsSpan().ToArray(), "dense rows");
        Compare(() => Fourier.ForwardColumns(tall).AsSpan().ToArray(), "columns");
        Compare(() => Fourier.InverseRows(padded[0..64, ..]).AsSpan().ToArray(), "view rows");
        Compare(() => Fourier.ForwardRealRows(real).AsSpan().ToArray(), "dense real rows");
        Compare(() => Fourier.ForwardRealRows(realPadded[0..64, ..]).AsSpan().ToArray(), "view real rows");
        Compare(() => Fourier.ForwardRealColumns(real.T).AsSpan().ToArray(), "real columns");
        var spectrum = Fourier.ForwardRealRows(real);
        Compare(() => Fourier.InverseRealRows(spectrum, 2050).AsSpan().ToArray(), "inverse real rows");
        Compare(() => Fourier.InverseRealColumns(Fourier.ForwardRealColumns(real.T), 2050).AsSpan().ToArray(), "inverse real columns");
        Compare(() => Fourier.CosineRows(square, 2).AsSpan().ToArray(), "cosine rows");
        Compare(() => Fourier.CosineColumns(square, 2).AsSpan().ToArray(), "cosine columns");
    }

    [Fact]
    public void ContinuousWaveletMatchesSingleThreadBitwise()
    {
        var x = RandomReal(4096, 7);
        var scales = Wavelets.GeometricScales(1, 0.25, 48);
        foreach (var padding in new[] { WaveletPadding.Zero, WaveletPadding.Periodic })
        {
            Compare(() => Wavelets.Continuous(x, scales, ContinuousWavelet.Morlet(), 1, padding).AsSpan().ToArray(), $"cwt {padding}");
        }
    }

    /// <summary>1スレッドと使えるすべてのスレッドで計算して、ビット単位で比べる（1コアの環境では比べるものがない）</summary>
    private static void Compare<T>(Func<T[]> compute, string label)
        where T : unmanaged
    {
        int threads = Environment.ProcessorCount;
        if (threads < 2)
        {
            return;
        }

        T[] single = WithThreads(1, compute);
        T[] parallel = WithThreads(threads, compute);
        Assert.Equal(single.Length, parallel.Length);
        var a = System.Runtime.InteropServices.MemoryMarshal.AsBytes(single.AsSpan());
        var b = System.Runtime.InteropServices.MemoryMarshal.AsBytes(parallel.AsSpan());
        int index = a.SequenceCompareTo(b) == 0 ? -1 : FirstDifference(a, b) / System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
        Assert.True(index < 0, $"{label}: element {index} differs between 1 and {threads} threads");
    }

    private static int FirstDifference(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b)
    {
        int i = 0;
        while (a[i] == b[i])
        {
            i++;
        }

        return i;
    }

    private static T WithThreads<T>(int threads, Func<T> action)
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
