using System.Numerics;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.NeuralNetwork.Tests.Training;

[Collection(nameof(ParallelismCollection))]
public class ConvolutionTrainingTests
{
    private static readonly Padding2D Padding = new(1, 0, 2, 1);

    [Fact]
    public void FloatConvolutionMatchesInferenceAndNumPy() => CheckReference<float>();

    [Fact]
    public void DoubleConvolutionMatchesInferenceAndNumPy() => CheckReference<double>();

    [Fact]
    public void AnEmptyOutputLeavesTheParameterGradientsUnchanged()
    {
        var tape = new Tape();
        var w = Parameter<double>.Matrix(2, 9);
        w.Gradient.Fill(3);
        var x = tape.Leaf(new Tensor<double>(1, 2, 2));
        var y = XMath.Convolve2D(x, tape.UseMatrix(w), 3, 3, default);
        Assert.Equal((2, 0, 0), (y.Rows, y.Columns, y.MatrixCount));
        tape.Backward(XMath.Sum(XMath.ColumnSums(XMath.SumMatrices(y))));
        Assert.All(w.Gradient.ToArray(), value => Assert.Equal(3d, value));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void ConvolutionTrainingReusesItsBuffers(int degree)
    {
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = degree;
            var tape = new Tape();
            var x = new Tensor<float>(16, 31, 29);
            var w = Parameter<float>.Matrix(32, 16 * 3 * 3);
            var b = Parameter<float>.Vector(32);
            var padding = new Padding2D(1, 1);
            x.AsSpan().Fill(0.25f);
            w.Value.Fill(-0.125f);
            for (int i = 0; i < 4; i++)
            {
                Step();
            }

            long bytes = -1;
            for (int window = 0; window < 3 && bytes != 0; window++)
            {
                GC.Collect(0, GCCollectionMode.Forced, blocking: true);
                long before = GC.GetAllocatedBytesForCurrentThread();
                for (int i = 0; i < 5; i++)
                {
                    Step();
                }

                bytes = GC.GetAllocatedBytesForCurrentThread() - before;
            }

            Assert.Equal(0, bytes);

            void Step()
            {
                tape.Reset();
                w.ZeroGradient();
                b.ZeroGradient();
                var y = XMath.Convolve2D(tape.Leaf(x), tape.UseMatrix(w), 3, 3, padding, tape.UseVector(b));
                tape.Backward(XMath.Sum(XMath.ColumnSums(XMath.SumMatrices(y))));
            }
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    [Fact]
    public void ContiguousInputTransposedWeightsAndNoBiasMatchReference()
    {
        var x = new Tensor<double>(2, 5, 4);
        Convert<double>(ConvolutionReferences.Input).CopyTo(x.AsSpan());
        var weights = Matrix<double>.FromColumnMajor(3, 12, ConvolutionReferences.PackedWeights);
        var bias = new LinearAlgebra.Vector<double>(ConvolutionReferences.Bias);
        var seed = Matrix<double>.FromColumnMajor(3, 4, ConvolutionReferences.Seed);

        // 非転置の連続入力 + 転置 weights ビュー
        {
            var tape = new Tape();
            var xNode = tape.Leaf(x);
            var wNode = tape.Leaf(weights.T.ToMatrix()).T;
            var bNode = tape.Leaf(bias);
            var y = XMath.Convolve2D(xNode, wNode, 2, 3, Padding, bNode, strideHeight: 2, strideWidth: 2, dilationWidth: 2);
            Assert.Equal(ConvolutionReferences.Output, y.Value.Data.ToArray());
            tape.Backward(XMath.Sum(XMath.ColumnSums(XMath.Hadamard(y.ToMatrix(), tape.Constant(seed)))));
            Assert.Equal(ConvolutionReferences.InputGradient, tape.GetGradient(xNode).ToTensor().AsSpan().ToArray());
            Assert.Equal(ConvolutionReferences.WeightGradient, tape.GetGradient(wNode).ToMatrix().AsSpan().ToArray());
            Assert.Equal(ConvolutionReferences.BiasGradient, tape.GetGradient(bNode).ToArray());
        }

        // bias 省略（default）でも入力・重み勾配は一致する
        {
            var tape = new Tape();
            var xNode = tape.Leaf(x);
            var wNode = tape.Leaf(weights);
            var y = XMath.Convolve2D(xNode, wNode, 2, 3, Padding, strideHeight: 2, strideWidth: 2, dilationWidth: 2);
            tape.Backward(XMath.Sum(XMath.ColumnSums(XMath.Hadamard(y.ToMatrix(), tape.Constant(seed)))));
            Assert.Equal(ConvolutionReferences.InputGradient, tape.GetGradient(xNode).ToTensor().AsSpan().ToArray());
            Assert.Equal(ConvolutionReferences.WeightGradient, tape.GetGradient(wNode).ToMatrix().AsSpan().ToArray());
        }

        // 行列軸スライス（FirstMatrix > 0）
        {
            var padded = new Tensor<double>(2, 5, 6);
            for (int h = 0; h < 4; h++)
            {
                x.GetMatrix(h).CopyTo(padded.GetMatrix(h + 1));
            }

            var tape = new Tape();
            var full = tape.Leaf(padded);
            var sliced = full[.., .., 1..5];
            var wNode = tape.Leaf(weights);
            var y = XMath.Convolve2D(sliced, wNode, 2, 3, Padding, strideHeight: 2, strideWidth: 2, dilationWidth: 2);
            tape.Backward(XMath.Sum(XMath.ColumnSums(XMath.Hadamard(y.ToMatrix(), tape.Constant(seed)))));
            var fullGrad = tape.GetGradient(full);
            Assert.All(fullGrad.GetMatrix(0).ToMatrix().AsSpan().ToArray(), v => Assert.Equal(0.0, v));
            Assert.All(fullGrad.GetMatrix(5).ToMatrix().AsSpan().ToArray(), v => Assert.Equal(0.0, v));
            Assert.Equal(ConvolutionReferences.InputGradient, fullGrad[.., .., 1..5].ToTensor().AsSpan().ToArray());
        }
    }

    [Fact]
    public void PartialGradientRequirementsAccumulationAndSpecialGeometriesWork()
    {
        var x = new Tensor<double>(2, 5, 4);
        Convert<double>(ConvolutionReferences.Input).CopyTo(x.AsSpan());
        var weights = Matrix<double>.FromColumnMajor(3, 12, ConvolutionReferences.PackedWeights);
        var bias = new LinearAlgebra.Vector<double>(ConvolutionReferences.Bias);
        var seed = Matrix<double>.FromColumnMajor(3, 4, ConvolutionReferences.Seed);

        // 全定数
        {
            var tape = new Tape();
            var y = XMath.Convolve2D(tape.Constant(x), tape.Constant(weights), 2, 3, Padding, tape.Constant(bias),
                strideHeight: 2, strideWidth: 2, dilationWidth: 2);
            Assert.False(y.RequiresGradient);
            Assert.Equal(ConvolutionReferences.Output, y.Value.Data.ToArray());
        }

        // x のみ / weights のみ / bias のみ勾配要、および既存勾配への蓄積
        {
            var tape = new Tape();
            var xNode = tape.Leaf(x);
            var y = XMath.Convolve2D(xNode, tape.Constant(weights), 2, 3, Padding, tape.Constant(bias),
                strideHeight: 2, strideWidth: 2, dilationWidth: 2);
            tape.Backward(XMath.Sum(XMath.ColumnSums(XMath.Hadamard(y.ToMatrix(), tape.Constant(seed)))));
            Assert.Equal(ConvolutionReferences.InputGradient, tape.GetGradient(xNode).ToTensor().AsSpan().ToArray());
        }

        {
            var tape = new Tape();
            var wParam = Parameter<double>.Matrix(3, 12);
            ConvolutionReferences.PackedWeights.CopyTo(wParam.Value);
            wParam.Gradient.Fill(2.0);
            var y = XMath.Convolve2D(tape.Constant(x), tape.UseMatrix(wParam), 2, 3, Padding, tape.Constant(bias),
                strideHeight: 2, strideWidth: 2, dilationWidth: 2);
            tape.Backward(XMath.Sum(XMath.ColumnSums(XMath.Hadamard(y.ToMatrix(), tape.Constant(seed)))));
            for (int i = 0; i < wParam.Gradient.Length; i++)
            {
                Assert.Equal(2.0 + ConvolutionReferences.WeightGradient[i], wParam.Gradient[i], 13);
            }
        }

        {
            var tape = new Tape();
            var bNode = tape.Leaf(bias);
            var y = XMath.Convolve2D(tape.Constant(x), tape.Constant(weights), 2, 3, Padding, bNode,
                strideHeight: 2, strideWidth: 2, dilationWidth: 2);
            tape.Backward(XMath.Sum(XMath.ColumnSums(XMath.Hadamard(y.ToMatrix(), tape.Constant(seed)))));
            Assert.Equal(ConvolutionReferences.BiasGradient, tape.GetGradient(bNode).ToArray());
        }

        // 1x1 カーネルと dilationHeight > 1 の中心差分検証
        {
            var x2 = new Tensor<double>(2, 4, 5);
            for (int i = 0; i < x2.AsSpan().Length; i++)
            {
                x2.AsSpan()[i] = ((i % 9) - 4) * 0.25;
            }

            var w2 = Matrix<double>.FromColumnMajor(2, 4, [0.5, -0.3, 0.2, 0.7, -0.4, 0.6, 0.1, -0.2]);
            var b2 = new LinearAlgebra.Vector<double>([0.1, -0.2]);
            var pad = new Padding2D(1, 0);
            var tape = new Tape();
            var xn = tape.Leaf(x2);
            var wn = tape.Leaf(w2);
            var bn = tape.Leaf(b2);
            var y2 = XMath.Convolve2D(xn, wn, 2, 1, pad, bn, strideHeight: 1, strideWidth: 1, dilationHeight: 2, dilationWidth: 1);
            tape.Backward(XMath.Sum(XMath.ColumnSums(y2.ToMatrix())));

            var dx = tape.GetGradient(xn);
            double h = 1e-6;
            double EvalLoss()
            {
                var t = new Tape();
                var res = XMath.Convolve2D(t.Constant(x2), t.Constant(w2), 2, 1, pad, t.Constant(b2),
                    strideHeight: 1, strideWidth: 1, dilationHeight: 2, dilationWidth: 1);
                return XMath.Sum(XMath.ColumnSums(res.ToMatrix())).Value;
            }

            double orig = x2[1, 2, 2];
            x2[1, 2, 2] = orig + h;
            double plus = EvalLoss();
            x2[1, 2, 2] = orig - h;
            double minus = EvalLoss();
            x2[1, 2, 2] = orig;
            Assert.Equal((plus - minus) / (2 * h), dx[1, 2, 2], 7);
        }

        // 空入力（width == 0）+ パディングで非空出力
        {
            var tape = new Tape();
            var emptyX = tape.Leaf(new Tensor<double>(2, 0, 2));
            var w1 = tape.Leaf(Matrix<double>.FromColumnMajor(3, 2, [1, 2, 3, 4, 5, 6]));
            var b1 = tape.Leaf(new LinearAlgebra.Vector<double>([0.5, -0.25, 1.0]));
            var y = XMath.Convolve2D(emptyX, w1, 1, 1, new Padding2D(0, 0, 1, 1), b1);
            Assert.Equal((3, 2, 2), (y.Rows, y.Columns, y.MatrixCount));
            tape.Backward(XMath.Sum(XMath.ColumnSums(y.ToMatrix())));
            Assert.All(tape.GetGradient(w1).ToMatrix().AsSpan().ToArray(), v => Assert.Equal(0.0, v));
            Assert.Equal(new[] { 4.0, 4.0, 4.0 }, tape.GetGradient(b1).ToArray());
        }
    }

    [Fact]
    public void ConvolutionValuesAndGradientsDoNotDependOnDegreeOfParallelism()
    {
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            var x = new Tensor<float>(8, 17, 15);
            var w = new Matrix<float>(12, 8 * 3 * 3);
            var b = new LinearAlgebra.Vector<float>(12);
            for (int i = 0; i < x.AsSpan().Length; i++)
            {
                x.AsSpan()[i] = ((i * 7) % 19 - 9) * 0.125f;
            }

            for (int i = 0; i < w.AsSpan().Length; i++)
            {
                w.AsSpan()[i] = ((i * 5) % 13 - 6) * 0.125f;
            }

            for (int i = 0; i < b.Length; i++)
            {
                b[i] = (i - 5) * 0.25f;
            }

            (float[] y, float[] dx, float[] dw, float[] db) Run(int degree)
            {
                LinearAlgebraSettings.MaxDegreeOfParallelism = degree;
                var tape = new Tape();
                var xn = tape.Leaf(x);
                var wn = tape.Leaf(w);
                var bn = tape.Leaf(b);
                var y = XMath.Convolve2D(xn, wn, 3, 3, new Padding2D(1, 0, 2, 1), bn, strideHeight: 2, strideWidth: 1, dilationHeight: 1, dilationWidth: 2);
                tape.Backward(XMath.Sum(XMath.ColumnSums(y.ToMatrix())));
                return (y.Value.Data.ToArray(), tape.GetGradient(xn).ToTensor().AsSpan().ToArray(),
                    tape.GetGradient(wn).ToMatrix().AsSpan().ToArray(), tape.GetGradient(bn).ToArray());
            }

            var single = Run(1);
            var parallel = Run(4);
            Assert.Equal(single.y, parallel.y);
            Assert.Equal(single.dx, parallel.dx);
            Assert.Equal(single.dw, parallel.dw);
            Assert.Equal(single.db, parallel.db);
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    [Fact]
    public void InvalidArgumentsAndTapeErrorsAreRejected()
    {
        var tape = new Tape();
        var other = new Tape();
        var x = tape.Leaf(new Tensor<double>(2, 4, 4));
        var w = tape.Leaf(new Matrix<double>(3, 2 * 3 * 3));
        var b = tape.Leaf(new LinearAlgebra.Vector<double>(3));

        // 未初期化ノードは ArgumentOutOfRangeException より優先して InvalidOperationException
        Assert.Throws<InvalidOperationException>(() => XMath.Convolve2D(default, w, 3, 3, default));
        Assert.Throws<InvalidOperationException>(() => XMath.Convolve2D(x, default, 3, 3, default));

        // 負パディング・0以下次元・形状不一致
        Assert.Throws<ArgumentOutOfRangeException>(() => XMath.Convolve2D(x, w, 3, 3, new Padding2D(-1, 0)));
        Assert.Throws<ArgumentOutOfRangeException>(() => XMath.Convolve2D(x, w, 0, 3, default));
        Assert.Throws<ArgumentOutOfRangeException>(() => XMath.Convolve2D(x, w, 3, 3, default, strideHeight: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => XMath.Convolve2D(x, w, 3, 3, default, dilationWidth: 0));
        Assert.Throws<ArgumentException>(() => XMath.Convolve2D(x, tape.Leaf(new Matrix<double>(3, 5)), 3, 3, default));
        Assert.Throws<ArgumentException>(() => XMath.Convolve2D(x, w, 3, 3, default, tape.Leaf(new LinearAlgebra.Vector<double>(2))));

        // 非対応要素型
        Assert.Throws<NotSupportedException>(() => XMath.Convolve2D(default(TensorNode<Half>), default, 1, 1, default));

        // 別テープ混在
        Assert.Throws<InvalidOperationException>(() => XMath.Convolve2D(x, other.Leaf(new Matrix<double>(3, 2 * 3 * 3)), 3, 3, default));

        // 例外後も同じテープで正常に記録・逆伝播できる
        var y = XMath.Convolve2D(x, w, 3, 3, default, b);
        tape.Backward(XMath.Sum(XMath.ColumnSums(y.ToMatrix())));
        Assert.Equal((3, 2, 2), (y.Rows, y.Columns, y.MatrixCount));

        // 逆伝播完了後の記録拒否
        Assert.Throws<InvalidOperationException>(() => XMath.Convolve2D(x, w, 3, 3, default, b));

        // Reset 後の失効ノード
        tape.Reset();
        Assert.Throws<InvalidOperationException>(() => XMath.Convolve2D(x, w, 3, 3, default, b));
    }

    private static void CheckReference<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        var x = new Tensor<T>(2, 5, 4);
        Convert<T>(ConvolutionReferences.Input).CopyTo(x.AsSpan());
        var weights = Parameter<T>.Matrix(3, 12);
        var bias = Parameter<T>.Vector(3);
        Convert<T>(ConvolutionReferences.PackedWeights).CopyTo(weights.Value);
        Convert<T>(ConvolutionReferences.Bias).CopyTo(bias.Value);
        var seed = new Matrix<T>(3, 4);
        Convert<T>(ConvolutionReferences.Seed).CopyTo(seed.AsSpan());

        var tape = new Tape();
        var input = tape.Leaf(x.T.ToTensor()).T;
        var result = XMath.Convolve2D(input, tape.UseMatrix(weights), 2, 3, Padding, tape.UseVector(bias),
            strideHeight: 2, strideWidth: 2, dilationWidth: 2);
        Assert.Equal((3, 2, 2), (result.Rows, result.Columns, result.MatrixCount));
        Assert.Equal(Convert<T>(ConvolutionReferences.Output), result.Value.Data.ToArray());

        var inference = new Convolution2D<T>(Convert<T>(ConvolutionReferences.InferenceWeights), bias.Value, 3, 2, 2, 3,
            strideHeight: 2, strideWidth: 2, dilationWidth: 2);
        Assert.True(inference.Forward(x, Padding).AsSpan().SequenceEqual(result.Value.Data));
        var loss = XMath.Sum(XMath.ColumnSums(XMath.Hadamard(result.ToMatrix(), tape.Constant(seed))));
        tape.Backward(loss);
        Assert.Equal(Convert<T>(ConvolutionReferences.InputGradient), tape.GetGradient(input).ToTensor().AsSpan().ToArray());
        Assert.Equal(Convert<T>(ConvolutionReferences.WeightGradient), weights.Gradient.ToArray());
        Assert.Equal(Convert<T>(ConvolutionReferences.BiasGradient), bias.Gradient.ToArray());

        T original = x[0, 2, 1], h = T.CreateChecked(0.125);
        x[0, 2, 1] = original + h;
        double plus = WeightedValue(inference, x, seed);
        x[0, 2, 1] = original - h;
        double minus = WeightedValue(inference, x, seed);
        Assert.Equal(ConvolutionReferences.InputGradient[(5 + 2) * 2], (plus - minus) / 0.25, 8);
    }

    private static double WeightedValue<T>(Convolution2D<T> layer, Tensor<T> x, Matrix<T> seed)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var output = layer.Forward(x, Padding);
        return double.CreateChecked(XMath.Sum(XMath.ColumnSums(XMath.Hadamard(output.AsMatrix(), seed))));
    }

    private static T[] Convert<T>(double[] values) where T : unmanaged, IFloatingPointIeee754<T> =>
        Array.ConvertAll(values, value => T.CreateChecked(value));
}
