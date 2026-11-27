using System.Numerics;
using Tuna.Differentiation;

namespace Tuna.Tests.Differentiation;

using Tuna.LinearAlgebra;

public class OuterProductTests
{
    [Fact]
    public void ViewsMatchCoreValuesAndScalarGradientSums()
    {
        CheckViews<double>();
        CheckViews<float>();
    }

    private static void CheckViews<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        var storage = Matrix<T>.Generate(3, 8, (i, j) => T.CreateChecked(3 * i - j));
        var sourceU = storage.T.Column(1);
        var sourceV = storage.Row(2);
        var tape = new Tape();
        var uLeaf = tape.Leaf(sourceU);
        var vLeaf = tape.Leaf(sourceV);
        var u = uLeaf[1..^1];
        var v = vLeaf[2..^2];
        var product = XMath.Outer(u, v);
        var expected = XMath.Outer(sourceU[1..^1], sourceV[2..^2]);
        ElementwiseOperations.AssertSameBits<T>(expected.AsSpan(), product.Value.ToMatrix().AsSpan(), typeof(T).Name);

        var storageWeights = Matrix<T>.Generate(v.Length, u.Length, (j, i) => T.CreateChecked(i + 1 - 2 * j));
        var weights = storageWeights.T;
        tape.Backward(Sum(XMath.Hadamard(product, tape.Constant(weights))));

        var du = tape.GetGradient(uLeaf);
        for (int i = 0; i < du.Length; i++)
        {
            T sum = T.Zero;
            if (i >= 1 && i < du.Length - 1)
            {
                for (int j = 0; j < v.Length; j++)
                {
                    sum += weights[i - 1, j] * sourceV[j + 2];
                }
            }

            Assert.Equal(sum, du[i]);
        }

        var dv = tape.GetGradient(vLeaf);
        for (int j = 0; j < dv.Length; j++)
        {
            T sum = T.Zero;
            if (j >= 2 && j < dv.Length - 2)
            {
                for (int i = 0; i < u.Length; i++)
                {
                    sum += weights[i, j - 2] * sourceU[i + 1];
                }
            }

            Assert.Equal(sum, dv[j]);
        }
    }

    [Fact]
    public void GradientsMatchFiniteDifferences()
    {
        var u = new Vector<double>([1.2, -0.4, 0.7]);
        var v = new Vector<double>([2.1, -3.5]);
        var weights = Matrix<double>.FromRowMajor(3, 2, [0.3, -1.2, 2.8, -0.5, 0.1, -2.0]);
        var tape = new Tape();
        var un = tape.Leaf(u);
        var vn = tape.Leaf(v);
        tape.Backward(Sum(XMath.Hadamard(XMath.Outer(un, vn), tape.Constant(weights))));

        Check(u, tape.GetGradient(un));
        Check(v, tape.GetGradient(vn));

        void Check(Vector<double> input, ReadOnlyVectorSpan<double> gradient)
        {
            for (int i = 0; i < input.Length; i++)
            {
                int index = i;
                double original = input[index];
                double expected = DifferentiationAssert.Richardson(value =>
                {
                    input[index] = value;
                    return XMath.Sum(XMath.ColumnSums(XMath.Hadamard(XMath.Outer(u, v), weights)));
                }, original);
                input[index] = original;
                DifferentiationAssert.Close(expected, gradient[index], 1e-8, 1e-8);
            }
        }
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(0, 3)]
    [InlineData(3, 0)]
    public void EmptyProductsPreserveShapeAndHaveZeroGradients(int m, int n)
    {
        CheckEmpty<double>(m, n);
        CheckEmpty<float>(m, n);
    }

    private static void CheckEmpty<T>(int m, int n) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var tape = new Tape();
        var u = tape.Leaf(new Vector<T>(m));
        var v = tape.Leaf(new Vector<T>(n));
        var product = XMath.Outer(u, v);
        Assert.Equal((m, n), (product.Rows, product.Columns));
        tape.Backward(Sum(product));
        Assert.Equal(new T[m], tape.GetGradient(u).ToArray());
        Assert.Equal(new T[n], tape.GetGradient(v).ToArray());
    }

    [Fact]
    public void SameNodeAndConstantOperandsAccumulateOnlyRequestedGradients()
    {
        CheckAccumulation<double>();
        CheckAccumulation<float>();
    }

    private static void CheckAccumulation<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        var tape = new Tape();
        var u = tape.Leaf(new Vector<T>([T.One, T.CreateChecked(2), T.CreateChecked(3)]));
        var v = tape.Constant(new Vector<T>([T.CreateChecked(-2), T.CreateChecked(5)]));
        var uu = XMath.Outer(u, u);
        var uv = XMath.Outer(u, v);
        var vu = XMath.Outer(v, u);
        Assert.False(XMath.Outer(v, v).RequiresGradient);
        tape.Backward(Sum(uu) + Sum(uv) + Sum(vu));
        Assert.Equal(Enumerable.Repeat(T.CreateChecked(18), u.Length), tape.GetGradient(u).ToArray());
    }

    [Fact]
    public void GemvContributionsPropagateProductsOfZeroAndInfinity()
    {
        CheckNonFinite<double>();
        CheckNonFinite<float>();
    }

    private static void CheckNonFinite<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        var tape = new Tape();
        var u = tape.Leaf(new Vector<T>([T.Zero, T.CreateChecked(2)]));
        var v = tape.Leaf(new Vector<T>([T.NegativeZero, T.CreateChecked(-3)]));
        var weights = tape.Constant(Matrix<T>.FromRowMajor(2, 2, [T.PositiveInfinity, T.CreateChecked(2), T.CreateChecked(3), T.CreateChecked(4)]));
        tape.Backward(Sum(XMath.Hadamard(XMath.Outer(u, v), weights)));
        var du = tape.GetGradient(u);
        var dv = tape.GetGradient(v);
        Assert.True(T.IsNaN(du[0]));
        Assert.True(T.IsNaN(dv[0]));
        Assert.Equal(T.CreateChecked(-12), du[1]);
        Assert.Equal(T.CreateChecked(8), dv[1]);
    }

    [Fact]
    public void UnreachableProductDoesNotContributeToGradients()
    {
        var tape = new Tape();
        var u = tape.Leaf(new Vector<double>([1, 2, 3]));
        var v = tape.Leaf(new Vector<double>([double.PositiveInfinity, double.NaN]));
        XMath.Outer(u, v);
        tape.Backward(XMath.Sum(u));
        Assert.Equal(new double[] { 1, 1, 1 }, tape.GetGradient(u).ToArray());
        Assert.Equal(new double[2], tape.GetGradient(v).ToArray());
    }

    private static ScalarNode Sum<T>(MatrixNode<T> matrix) where T : unmanaged, IFloatingPointIeee754<T>
        => XMath.Sum(XMath.ColumnSums(matrix));
}
