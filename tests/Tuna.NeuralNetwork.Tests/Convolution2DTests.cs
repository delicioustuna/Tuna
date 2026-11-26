using System.Numerics;

namespace Tuna.NeuralNetwork.Tests;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

/// <summary>2次元畳み込みを nn.Conv2d の定義と比べ、ストリーム版が因果的な詰め方の一括版と同じビットになることを確かめる</summary>
public class Convolution2DTests
{
    public static TheoryData<int, int, int, int, int, int, int, int, int, int> Cases => new()
    {
        // inC, outC, kh, kw, sh, sw, dh, dw, width, height
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 3, 5, 3, 3, 1, 1, 1, 1, 9, 6 },
        { 4, 8, 1, 3, 1, 2, 1, 1, 17, 3 },   // 幅方向に間隔 2（RE-USE のエンコーダの2つめの畳み込み）
        { 6, 4, 3, 3, 1, 1, 4, 1, 7, 12 },   // 高さ方向の膨張（DenseBlock）
        { 2, 3, 2, 4, 2, 3, 2, 2, 15, 9 },
        { 64, 32, 3, 1, 1, 1, 1, 1, 5, 4 },  // 高さ方向だけの核
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void ForwardMatchesDefinition(int inC, int outC, int kh, int kw, int sh, int sw, int dh, int dw, int width, int height)
    {
        Forward<float>(inC, outC, kh, kw, sh, sw, dh, dw, width, height);
        Forward<double>(inC, outC, kh, kw, sh, sw, dh, dw, width, height);
    }

    private static void Forward<T>(int inC, int outC, int kh, int kw, int sh, int sw, int dh, int dw, int width, int height)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var w = TestData.Rounded<T>(TestData.Uniform(outC * inC * kh * kw, 70, 1 / Math.Sqrt(inC * kh * kw)));
        var b = TestData.Rounded<T>(TestData.Uniform(outC, 71));
        var x = TestData.Rounded<T>(TestData.Uniform(inC * width * height, 72));
        var conv = new Convolution2D<T>(TestData.To<T>(w), TestData.To<T>(b), outC, inC, kh, kw, sh, sw, dh, dw);
        var input = Tensor<T>.FromColumnMajor([inC, width, height], TestData.To<T>(x));
        foreach (var padding in new[] { new Padding2D(0, 0), new Padding2D(1, 2), new Padding2D(2, 0, 1, 3) })
        {
            var expected = Reference.Convolution2D(x, inC, width, height, w, b, outC, kh, kw, sh, sw, dh, dw, padding, out int ow, out int oh);
            Assert.Equal(ow, conv.OutputWidth(width, padding));
            Assert.Equal(oh, conv.OutputHeight(height, padding));
            var actual = conv.Forward(input, padding);
            Assert.Equal(new nint[] { outC, ow, oh }, actual.Lengths.ToArray());
            TestData.Close<T>(expected, actual, TestData.Tolerance<T>());
        }

        // バイアスなし
        var noBias = new Convolution2D<T>(TestData.To<T>(w), default, outC, inC, kh, kw, sh, sw, dh, dw);
        TestData.Close<T>(Reference.Convolution2D(x, inC, width, height, w, null, outC, kh, kw, sh, sw, dh, dw, new Padding2D(1, 1), out _, out _),
            noBias.Forward(input, new Padding2D(1, 1)), TestData.Tolerance<T>());
    }

    [Fact]
    public void StridedInputAndOutputGiveTheSameBits()
    {
        Strided<float>();
        Strided<double>();
    }

    private static void Strided<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        const int inC = 5, outC = 6, width = 8, height = 7;
        var conv = new Convolution2D<T>(TestData.To<T>(TestData.Uniform(outC * inC * 9, 80)), TestData.To<T>(TestData.Uniform(outC, 81)),
            outC, inC, 3, 3);
        var input = Tensor<T>.FromColumnMajor([inC, width, height], TestData.To<T>(TestData.Uniform(inC * width * height, 82)));
        var padding = new Padding2D(1, 1);
        var expected = conv.Forward(input, padding);

        // 大きなテンソルの一部（チャネルの間隔が 1、位置の間隔が大きい）
        var larger = new Tensor<T>(inC + 3, width + 2, height);
        input.AsTensorSpan().CopyTo(larger[1..(inC + 1), 2..(width + 2), ..]);
        TestData.Same<T>(expected, conv.Forward(larger[1..(inC + 1), 2..(width + 2), ..], padding));

        // チャネルが連続しない入力と出力先（行と列を入れ替えたテンソルの転置ビュー）
        var swapped = new Tensor<T>(width, inC, height);
        input.AsTensorSpan().CopyTo(swapped.T);
        var destination = new Tensor<T>(width, outC, height);
        conv.Forward(swapped.T, padding, destination.T);
        TestData.Same<T>(expected, destination.T);
    }

    [Theory]
    [InlineData(3, 4, 3, 3, 1, 1, 10, 6)]
    [InlineData(8, 8, 3, 3, 1, 4, 9, 20)]  // 高さ方向の膨張 4（過去 8 行）
    [InlineData(4, 6, 1, 3, 2, 1, 17, 3)]  // 高さ方向の核が 1（過去を持たない）
    [InlineData(6, 3, 3, 1, 1, 1, 5, 7)]
    public void StreamMatchesCausalForward(int inC, int outC, int kh, int kw, int sw, int dh, int width, int rows)
    {
        Stream<float>(inC, outC, kh, kw, sw, dh, width, rows);
        Stream<double>(inC, outC, kh, kw, sw, dh, width, rows);
    }

    private static void Stream<T>(int inC, int outC, int kh, int kw, int sw, int dh, int width, int rows)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var conv = new Convolution2D<T>(TestData.To<T>(TestData.Uniform(outC * inC * kh * kw, 90)), TestData.To<T>(TestData.Uniform(outC, 91)),
            outC, inC, kh, kw, 1, sw, dh, 1);
        var input = Tensor<T>.FromColumnMajor([inC, width, rows], TestData.To<T>(TestData.Uniform(inC * width * rows, 92)));
        int left = kw / 2, right = kw - 1 - kw / 2;
        var expected = conv.Forward(input, new Padding2D((kh - 1) * dh, 0, left, right));
        var stream = conv.CreateStream(width, left, right);
        Assert.Equal((int)expected.Lengths[1], stream.OutputWidth);
        var output = new Matrix<T>(outC, stream.OutputWidth);
        for (int pass = 0; pass < 2; pass++)
        {
            for (int t = 0; t < rows; t++)
            {
                stream.Push(input.GetMatrix(t), output);
                TestData.Same<T>(expected.GetMatrix(t), output);
            }

            // 最初の行の前に戻して、もう一度同じ結果
            stream.Reset();
        }
    }

    [Fact]
    public void StreamAcceptsStridedAndOverlappingRows()
    {
        StreamRows<float>();
        StreamRows<double>();
    }

    /// <summary>行の間隔が 1 でない入力の行、出力先と同じ入力の行でも、連続な行と同じビット</summary>
    private static void StreamRows<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        const int channels = 6, width = 11, rows = 9;
        var conv = new Convolution2D<T>(TestData.To<T>(TestData.Uniform(channels * channels * 9, 95)), TestData.To<T>(TestData.Uniform(channels, 96)),
            channels, channels, 3, 3, 1, 1, 2, 1);
        var input = Tensor<T>.FromColumnMajor([channels, width, rows], TestData.To<T>(TestData.Uniform(channels * width * rows, 97)));
        var reference = conv.CreateStream(width, 1, 1);
        var strided = conv.CreateStream(width, 1, 1);
        var inPlace = conv.CreateStream(width, 1, 1);
        var expected = new Matrix<T>(channels, width);
        var transposed = new Matrix<T>(width, channels);
        var output = new Matrix<T>(channels, width);
        for (int t = 0; t < rows; t++)
        {
            reference.Push(input.GetMatrix(t), expected);
            input.GetMatrix(t).CopyTo(transposed.T);
            strided.Push(transposed.T, output);
            TestData.Same<T>(expected, output);
            var row = input.GetMatrix(t).ToMatrix();
            inPlace.Push(row, row);
            TestData.Same<T>(expected, row);
        }
    }

    [Fact]
    public void InvalidArgumentsThrow()
    {
        Assert.Throws<ArgumentException>(() => new Convolution2D<float>(new float[10], default, 2, 1, 2, 2));
        Assert.Throws<ArgumentException>(() => new Convolution2D<float>(new float[8], new float[3], 2, 1, 2, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Convolution2D<float>(new float[8], default, 2, 1, 2, 2, strideHeight: 0));
        Assert.Throws<NotSupportedException>(() => new Convolution2D<Half>(new Half[1], default, 1, 1, 1, 1));
        var conv = new Convolution2D<float>(new float[8], default, 2, 1, 2, 2);
        Assert.Throws<ArgumentException>(() => conv.Forward(new Tensor<float>(2, 3, 3), default));
        Assert.Throws<ArgumentException>(() => conv.Forward(new Tensor<float>(1, 3, 3), default, new Tensor<float>(2, 3, 3)));
        Assert.Throws<ArgumentOutOfRangeException>(() => conv.Forward(new Tensor<float>(1, 3, 3), new Padding2D(-1, 0)));
        Assert.Throws<ArgumentException>(() => new Convolution2D<float>(new float[8], default, 2, 1, 2, 2, strideHeight: 2).CreateStream(3, 0, 0));
        var stream = conv.CreateStream(3, 0, 0);
        Assert.Throws<ArgumentException>(() => stream.Push(new Matrix<float>(1, 4), new Matrix<float>(2, 2)));
        Assert.Throws<ArgumentException>(() => stream.Push(new Matrix<float>(1, 3), new Matrix<float>(2, 3)));
    }
}
