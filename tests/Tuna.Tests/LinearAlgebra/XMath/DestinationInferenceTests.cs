using Tuna.LinearAlgebra;

namespace Tuna.Tests.LinearAlgebra;

public class DestinationInferenceTests
{
    [Fact]
    public void TransposedAndSlicedInputsReturnTheVectorDestination()
    {
        var a = new Matrix<double>(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var x = new Vector<double>([2, -1]);
        var destination = new Vector<double>(3);

        Assert.Same(destination, XMath.Multiply(a.T, x, destination));
        Assert.Equal(new double[] { -2, -1, 0 }, destination.ToArray());
        Assert.Same(destination, XMath.Multiply(a[.., ..].T, x.AsVectorSpan(), destination));
        ReadOnlyMatrixSpan<double> readOnly = a.T;
        Assert.Same(destination, XMath.Multiply(readOnly, x, destination));

        var storage = new Vector<double>(5);
        var viewResult = XMath.Multiply(a.T, x, storage[1..4]);
        Assert.Equal(new double[] { -2, -1, 0 }, viewResult.ToArray());
        Assert.Equal(new double[] { 0, -2, -1, 0, 0 }, storage.ToArray());
        Assert.Throws<ArgumentException>(() => XMath.Multiply(a.T, x, new Vector<double>(2)));
        Assert.Throws<ArgumentException>(() => XMath.Multiply(a.T, a.Column(0), a.Row(0)));
    }

    [Fact]
    public void MixedMatrixAndTensorInputsReturnTheContainerDestination()
    {
        var a = new Matrix<double>(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var matrixDestination = new Matrix<double>(3, 3);
        var expected = XMath.Multiply(a.T, a.AsMatrixSpan());
        Assert.Same(matrixDestination, XMath.Multiply(a.T, a, matrixDestination));
        Assert.Equal(expected.ToArray(), matrixDestination.ToArray());
        Assert.Same(matrixDestination, XMath.Multiply(a.T.ToMatrix(), a, matrixDestination));
        Assert.Equal(expected.ToArray(), matrixDestination.ToArray());

        var tensor = Tensor<double>.FromMatrices(a, 2.0 * a);
        var tensorDestination = new Tensor<double>(3, 3, 2);
        Assert.Same(tensorDestination, XMath.Multiply(tensor.T, tensor, tensorDestination));
        Assert.Equal(expected.ToArray(), tensorDestination.GetMatrix(0).ToArray());
        Assert.Equal((4.0 * expected).ToArray(), tensorDestination.GetMatrix(1).ToArray());
    }

    [Fact]
    public void BroadcastAndGeneratedUserOperationsAcceptContainerDestinations()
    {
        var x = new Vector<double>([1, 2]);
        var a = new Matrix<double>(new double[,] { { 10, 20 }, { 30, 40 } });
        var destination = new Matrix<double>(2, 2);
        Assert.Same(destination, XMath.Add(a, x.Broadcast(2), destination));
        Assert.Equal(new double[,] { { 11, 21 }, { 32, 42 } }, destination.ToArray());

        ReadOnlyVectorSpan<double> input = x;
        var output = new Vector<double>(2);
        Assert.Same(output, Activations.Square(input, output));
        Assert.Equal(new double[] { 1, 4 }, output.ToArray());
        Assert.Same(output, Activations.Square(x, output));
        Assert.Equal(new double[] { 1, 4 }, output.ToArray());
    }
}
