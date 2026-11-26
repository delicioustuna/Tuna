using Tuna;
using Tuna.LinearAlgebra;
using Tuna.NeuralNetwork;
using Xunit;

namespace Tuna.NeuralNetwork.Tests;

/// <summary>LayerNormの列長境界・固定参照と、Mambaの重複拒否時の状態保持を確認する</summary>
public class NormalizationAndStateBoundaryTests
{
    [Theory]
    [InlineData(31)]
    [InlineData(257)]
    [InlineData(1023)]
    [InlineData(4097)]
    [InlineData(16383)]
    [InlineData(65537)]
    [InlineData(100003)]
    public void ConstantSingleAtAdditionalColumnLengths(int length)
    {
        var x = new Matrix<float>(length, 1);
        foreach (float value in new[] { 1000.1f, 1.1f })
        {
            x.AsSpan().Fill(value);
            var y = XMath.LayerNorm(x, 1e-5f);
            foreach (float actual in y.AsSpan())
            {
                Assert.True(Math.Abs(actual) <= 2e-5f,
                    $"n={length}, x={value:R}, expected0, actual={actual:R}");
            }
        }
    }

    [Theory]
    [InlineData(31)]
    [InlineData(257)]
    [InlineData(1023)]
    [InlineData(4097)]
    [InlineData(16383)]
    [InlineData(65537)]
    [InlineData(100003)]
    public void ConstantDoubleAtAdditionalColumnLengths(int length)
    {
        var x = new Matrix<double>(length, 1);
        foreach (double value in new[] { 1000.1, 1.1 })
        {
            x.AsSpan().Fill(value);
            var y = XMath.LayerNorm(x, 1e-5);
            foreach (double actual in y.AsSpan())
            {
                Assert.True(Math.Abs(actual) <= 2e-12,
                    $"n={length}, x={value:R}, expected0, actual={actual:R}");
            }
        }
    }

    // Fixed high-precision references for x_i=1024+(-1)^i/8 and epsilon=double1e-5.
    // Rounded float epsilon differs negligibly within the existing2e-5 float budget.
    [Theory]
    [InlineData(1023, 0.9987034259122997, -1.0006578357477445)]
    [InlineData(1025, 0.9987053307945534, -1.0006559271437616)]
    public void BinaryOffsetWithSmallVariation(int length, double positive, double negative)
    {
        var x = new Matrix<double>(length, 1);
        var f = new Matrix<float>(length, 1);
        for (int i = 0; i < length; i++)
        {
            x[i, 0] = f[i, 0] = 1024 + (i % 2 == 0 ? 0.125f : -0.125f);
        }

        var yd = XMath.LayerNorm(x, 1e-5);
        var yf = XMath.LayerNorm(f, 1e-5f);
        for (int i = 0; i < length; i++)
        {
            double expected = i % 2 == 0 ? positive : negative;
            Assert.True(Math.Abs(yd[i, 0] - expected) <= 2e-12, $"double n={length}, i={i}: expected{expected:R}, actual{yd[i, 0]:R}");
            Assert.True(Math.Abs(yf[i, 0] - expected) <= 2e-5, $"float n={length}, i={i}: expected{expected:R}, actual{yf[i, 0]:R}");
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RejectedOverlapPreservesStateAndBuffers(bool forward)
    {
        const int model = 2, inner = 4, kernel = 3, sequences = 4;
        var wi = new float[2 * inner * model];
        for (int r = 0; r < 2 * inner; r++)
        {
            wi[r * model + r % model] = 1;
        }

        var wo = new float[model * inner];
        for (int r = 0; r < model; r++)
        {
            wo[r * inner + r] = 1;
        }

        var xProjection = new float[3 * inner];
        xProjection[inner] = xProjection[2 * inner] = 1;
        var block = new MambaBlock<float>(new(model, inner, 1, kernel, 1), wi,
            Enumerable.Repeat(1f, inner * kernel).ToArray(), [], xProjection,
            new float[inner], new float[inner], new float[inner],
            Enumerable.Repeat(1f, inner).ToArray(), wo);
        var input = new Matrix<float>(model, sequences);
        input.AsSpan().Fill(0.25f);
        var shared = new Matrix<float>(model, sequences + 1);
        shared.AsSpan().Fill(0.5f);
        var actualState = block.CreateState(forward ? 1 : sequences);
        var controlState = block.CreateState(forward ? 1 : sequences);
        var actual = new Matrix<float>(model, sequences);
        var control = new Matrix<float>(model, sequences);
        if (forward)
        {
            block.Forward(input, actualState, false, actual);
            block.Forward(input, controlState, false, control);
            Assert.Throws<ArgumentException>(() => block.Forward(shared[.., ..sequences], actualState, false, shared[.., 1..]));
            block.Forward(input, actualState, false, actual);
            block.Forward(input, controlState, false, control);
        }
        else
        {
            block.Step(input, actualState, false, actual);
            block.Step(input, controlState, false, control);
            Assert.Throws<ArgumentException>(() => block.Step(shared[.., ..sequences], actualState, false, shared[.., 1..]));
            block.Step(input, actualState, false, actual);
            block.Step(input, controlState, false, control);
        }

        Assert.All(shared.AsSpan().ToArray(), v => Assert.Equal(0.5f, v));
        for (int j = 0; j < sequences; j++)
        {
            for (int i = 0; i < model; i++)
            {
                Assert.Equal(BitConverter.SingleToInt32Bits(control[i, j]), BitConverter.SingleToInt32Bits(actual[i, j]));
            }
        }
    }
}
