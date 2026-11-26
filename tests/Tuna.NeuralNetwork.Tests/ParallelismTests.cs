using Tuna.LinearAlgebra;

namespace Tuna.NeuralNetwork.Tests;

/// <summary>
/// 並列の数（<see cref="LinearAlgebraSettings.MaxDegreeOfParallelism"/>）を変えても、ビット単位で同じ結果になることを確かめる。
/// 大きさは並列化される（演算量が並列の単位の数倍ある）ように選ぶ
/// </summary>
[Collection(nameof(ParallelismCollection))]
public class ParallelismTests
{
    private static readonly int[] Threads = [1, 2, 3, 4, 8];

    private static void SameForAllThreads(Func<Matrix<float>> compute)
    {
        var expected = TestData.WithThreads(1, compute);
        foreach (int threads in Threads.Skip(1))
        {
            TestData.Same<float>(expected, TestData.WithThreads(threads, compute));
        }
    }

    [Fact]
    public void NormalizationAndLinear()
    {
        const int channels = 64, positions = 4500; // PReLU（1要素1回）でも並列の単位の2倍を超える
        var x = TestData.Matrix<float>(TestData.Uniform(channels * positions, 200, 3), channels, positions);
        var scale = TestData.Vector<float>(TestData.Uniform(channels, 201, 0.5, 1.5));
        var shift = TestData.Vector<float>(TestData.Uniform(channels, 202));
        var w = TestData.Matrix<float>(TestData.Uniform(channels * channels, 203, 0.1), channels, channels);
        SameForAllThreads(() => XMath.LayerNorm(x, scale, shift, 1e-5f));
        SameForAllThreads(() => XMath.RmsNorm(x, scale, 1e-5f));
        SameForAllThreads(() => XMath.Prelu(x, scale));
        SameForAllThreads(() => XMath.Linear(w, x, shift));
    }

    [Fact]
    public void ConvolveAndScan()
    {
        const int channels = 256, states = 16, positions = 600;
        var x = TestData.Matrix<float>(TestData.Uniform(channels * positions, 210), channels, positions);
        var delta = TestData.Matrix<float>(TestData.Uniform(channels * positions, 211, 0.001, 0.5), channels, positions);
        var a = TestData.Matrix<float>(TestData.Uniform(channels * states, 212, -8, -0.5), channels, states);
        var b = TestData.Matrix<float>(TestData.Uniform(states * positions, 213), states, positions);
        var c = TestData.Matrix<float>(TestData.Uniform(states * positions, 214), states, positions);
        var d = TestData.Vector<float>(TestData.Uniform(channels, 215));
        var kernel = TestData.Matrix<float>(TestData.Uniform(channels * 4, 216), channels, 4);
        SameForAllThreads(() => XMath.CausalDepthwiseConvolve(x, kernel, d));
        SameForAllThreads(() => XMath.SelectiveScan(x, delta, a, b, c, d, approximate: true));
        SameForAllThreads(() => XMath.SelectiveScan(x, delta, a, b, c, d, approximate: false));
        SameForAllThreads(() =>
        {
            var state = new Tensor<float>(channels, states, positions);
            var y = new Matrix<float>(channels, positions);
            XMath.SelectiveScanStep<float>(x, delta, a, b, c, d, state, true, y);
            return y;
        });
    }

    [Fact]
    public void Convolution2DAndStream()
    {
        const int inC = 64, outC = 64, width = 130, height = 6;
        var conv = new Convolution2D<float>(TestData.To<float>(TestData.Uniform(outC * inC * 9, 220, 0.05)),
            TestData.To<float>(TestData.Uniform(outC, 221)), outC, inC, 3, 3, 1, 1, 2, 1);
        var input = Tensor<float>.FromColumnMajor([inC, width, height], TestData.To<float>(TestData.Uniform(inC * width * height, 222)));
        SameForAllThreads(() => conv.Forward(input, new Padding2D(4, 0, 1, 1)).AsMatrix().ToMatrix());
        SameForAllThreads(() =>
        {
            var stream = conv.CreateStream(width, 1, 1);
            var output = new Matrix<float>(outC, width * height);
            for (int t = 0; t < height; t++)
            {
                stream.Push(input.GetMatrix(t), output[.., (t * width)..((t + 1) * width)]);
            }

            return output;
        });
    }

    [Fact]
    public void MambaForwardAndStep()
    {
        var dimensions = MambaDimensions.Default(64);
        var block = new MambaWeights(dimensions, 230).Create<float>();
        const int length = 129;
        var u = TestData.Matrix<float>(TestData.Uniform(dimensions.Model * length, 231), dimensions.Model, length);
        SameForAllThreads(() => block.Forward(u, approximate: true));
        SameForAllThreads(() =>
        {
            var state = block.CreateState(length);
            var y = new Matrix<float>(dimensions.Model, length);
            for (int t = 0; t < 3; t++)
            {
                block.Step(u, state, true, y);
            }

            return y;
        });
    }
}
