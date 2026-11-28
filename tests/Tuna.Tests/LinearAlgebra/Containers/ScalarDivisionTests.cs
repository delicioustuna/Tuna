using System.Numerics;
using System.Runtime.InteropServices;

namespace Tuna.Tests.LinearAlgebra;

using Tuna.LinearAlgebra;

public class ScalarDivisionTests
{
    [Fact]
    public void OperatorsMatchDivideAndCompoundAssignmentPreservesIdentity()
    {
        const double divisor = 3;
        double[] values = [5, -7, 0.0, -0.0, double.Epsilon, double.NaN,
            double.PositiveInfinity, double.NegativeInfinity, double.MaxValue, 6, -10, 11];
        var vector = new Vector<double>(values);
        var matrix = Matrix<double>.FromColumnMajor(4, 3, values);
        var tensor = Tensor<double>.FromColumnMajor([2, 3, 2], values);

        var dividedVector = vector / divisor;
        var dividedMatrix = matrix / divisor;
        var dividedTensor = tensor / divisor;
        Assert.NotSame(vector, dividedVector);
        Assert.NotSame(matrix, dividedMatrix);
        Assert.NotSame(tensor, dividedTensor);
        AssertBits<double>(XMath.Divide(vector, divisor).AsSpan(), dividedVector.AsSpan());
        AssertBits<double>(XMath.Divide(matrix, divisor).AsSpan(), dividedMatrix.AsSpan());
        AssertBits<double>(XMath.Divide(tensor, divisor).AsSpan(), dividedTensor.AsSpan());

        CheckVectorView(vector[1..^1], divisor);
        CheckVectorView(matrix.Row(1), divisor);
        CheckVectorView(matrix.Column(0), divisor);
        CheckVectorView(matrix.Diagonal(), divisor);

        var submatrix = matrix.T[1.., ..3];
        var expectedMatrix = XMath.Divide(submatrix, divisor);
        AssertBits<double>(expectedMatrix.AsSpan(), (submatrix / divisor).AsSpan());
        AssertBits<double>(expectedMatrix.AsSpan(), (submatrix.AsReadOnly() / divisor).AsSpan());
        var broadcastMatrix = vector.BroadcastRows(3);
        AssertBits<double>(XMath.Divide(broadcastMatrix, divisor).AsSpan(), (broadcastMatrix / divisor).AsSpan());

        var subtensor = tensor.T[1.., .., ..];
        var expectedTensor = XMath.Divide(subtensor, divisor);
        AssertBits<double>(expectedTensor.AsSpan(), (subtensor / divisor).AsSpan());
        AssertBits<double>(expectedTensor.AsSpan(), (subtensor.AsReadOnly() / divisor).AsSpan());
        var broadcastTensor = matrix.Broadcast(3);
        AssertBits<double>(XMath.Divide(broadcastTensor, divisor).AsSpan(), (broadcastTensor / divisor).AsSpan());

        AssertBits<double>(values, vector.AsSpan());
        AssertBits<double>(values, matrix.AsSpan());
        AssertBits<double>(values, tensor.AsSpan());
        var vectorAlias = vector;
        var matrixAlias = matrix;
        var tensorAlias = tensor;
        vector /= divisor;
        matrix /= divisor;
        tensor /= divisor;
        Assert.Same(vectorAlias, vector);
        Assert.Same(matrixAlias, matrix);
        Assert.Same(tensorAlias, tensor);
        AssertBits<double>(dividedVector.AsSpan(), vector.AsSpan());
        AssertBits<double>(dividedMatrix.AsSpan(), matrix.AsSpan());
        AssertBits<double>(dividedTensor.AsSpan(), tensor.AsSpan());
    }

    private static void CheckVectorView(VectorSpan<double> view, double divisor)
    {
        var expected = XMath.Divide(view, divisor);
        AssertBits<double>(expected.AsSpan(), (view / divisor).AsSpan());
        AssertBits<double>(expected.AsSpan(), (view.AsReadOnly() / divisor).AsSpan());
    }

    [Fact]
    public void TensorScalarDivisionInfersElementTypeAndReturnsDestinationContainer()
    {
        var x = Tensor<double>.FromColumnMajor([1, 1, 1], [6.0]);
        var destination = new Tensor<double>(1, 1, 1);

        Assert.Equal(3.0, XMath.Divide(x, 2)[0, 0, 0]);
        Tensor<double> written = XMath.Divide(x, 2, destination);
        Assert.Same(destination, written);
        Assert.Equal(3.0, written[0, 0, 0]);
        Assert.Same(destination, XMath.Divide(x.AsTensorSpan().AsReadOnly(), 3, destination));
        Assert.Equal(2.0, destination[0, 0, 0]);
        XMath.InPlace.Divide(x, 2);
        Assert.Equal(3.0, x[0, 0, 0]);

        var single = Tensor<float>.FromColumnMajor([1, 1, 1], [6.0f]);
        Assert.Equal(3.0f, XMath.Divide(single, 2)[0, 0, 0]);
        Assert.Equal(3.0f, XMath.Divide(single.AsTensorSpan(), 2)[0, 0, 0]);

        var complex = Tensor<Complex>.FromColumnMajor([1, 1, 1], [new Complex(6, 8)]);
        Assert.Equal(new Complex(3, 4), XMath.Divide(complex, 2.0)[0, 0, 0]);
    }

    [Fact]
    public void TensorDivisionMatchesScalarArithmeticInDenseAndStridedLayouts()
    {
        CheckTensorDivision<double>();
        CheckTensorDivision<float>();
    }

    private static void CheckTensorDivision<T>() where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
    {
        const int rows = 17, columns = 3, count = 2;
        T[] samples = [T.CreateChecked(5), T.CreateChecked(-7), T.Zero, T.NegativeZero, T.Epsilon, -T.Epsilon,
            T.NaN, T.PositiveInfinity, T.NegativeInfinity, T.MaxValue, T.MinValue];
        var numerators = Enumerable.Range(0, rows * columns * count).Select(i => samples[i % samples.Length]).ToArray();
        var denominators = Enumerable.Range(0, numerators.Length).Select(i => samples[(3 * i + 2) % samples.Length]).ToArray();
        var x = Tensor<T>.FromColumnMajor([rows, columns, count], numerators);
        var y = Tensor<T>.FromColumnMajor([rows, columns, count], denominators);
        var denseOutput = new Tensor<T>(rows, columns, count);
        var strided = new TensorSpan<T>(new T[220], [rows, columns, count], [2, 35, 110]);
        var stridedOutput = new TensorSpan<T>(new T[220], [rows, columns, count], [2, 35, 110]);
        x.AsTensorSpan().AsReadOnly().CopyTo(strided);

        T[] divisors = [T.CreateChecked(3), T.Zero, T.NegativeZero, T.Epsilon, T.PositiveInfinity, T.NegativeInfinity, T.NaN];
        foreach (T divisor in divisors)
        {
            var expected = numerators.Select(v => v / divisor).ToArray();
            AssertBits<T>(expected, XMath.Divide(x, divisor).AsSpan());
            XMath.Divide(x, divisor, denseOutput);
            AssertBits<T>(expected, denseOutput.AsSpan());
            var written = XMath.Divide(strided, divisor, stridedOutput);
            AssertBits<T>(expected, written.ToTensor().AsSpan());
            AssertBits<T>(expected, stridedOutput.ToTensor().AsSpan());
        }

        var expectedElementwise = numerators.Zip(denominators, (a, b) => a / b).ToArray();
        AssertBits<T>(expectedElementwise, XMath.Divide(x, y).AsSpan());
        XMath.Divide(x, y, denseOutput);
        AssertBits<T>(expectedElementwise, denseOutput.AsSpan());
        XMath.Divide(strided, y.AsTensorSpan().AsReadOnly(), stridedOutput);
        AssertBits<T>(expectedElementwise, stridedOutput.ToTensor().AsSpan());

        T scale = T.CreateChecked(3);
        XMath.InPlace.Divide(strided, scale);
        AssertBits<T>(numerators.Select(v => v / scale).ToArray(), strided.ToTensor().AsSpan());
    }

    [Fact]
    public void TensorDivisionValidatesShapesAndOverlapBeforeWriting()
    {
        var x = Tensor<int>.FromColumnMajor([2, 2, 3], [8, -9, 12, 14, 18, -20, 33, 44, 50, 60, 71, 81]);
        var y = Tensor<int>.FromColumnMajor([2, 2, 2], [2, 2, -3, 3, 4, 4, 5, -6]);
        var original = x.Clone();

        Assert.Throws<ArgumentException>(() => XMath.Divide(x, y));
        Assert.Throws<ArgumentException>(() => XMath.Divide(x, x, y));
        Assert.Throws<ArgumentException>(() => XMath.Divide(x, 2, y));
        Assert.Throws<ArgumentException>("destination", () => XMath.Divide(x[.., .., ..2], 2, x[.., .., 1..]));
        Assert.Throws<ArgumentException>("destination", () => XMath.Divide(x[.., .., ..2], y, x[.., .., 1..]));
        Assert.Throws<ArgumentException>("destination", () => XMath.Divide(y, x[.., .., ..2], x[.., .., 1..]));
        Assert.Throws<ArgumentException>("destination", () => XMath.Divide(x.GetMatrix(0).Broadcast(3), 2, x));
        Assert.Equal(original.AsSpan(), x.AsSpan());

        var expected = XMath.Divide(x[.., .., ..2], y);
        var left = x[.., .., ..2];
        XMath.Divide(left, y, left);
        Assert.Equal(expected.AsSpan(), left.ToTensor().AsSpan());
        XMath.Divide(original[.., .., ..2], y, y);
        Assert.Equal(expected.AsSpan(), y.AsSpan());
    }

    [Fact]
    public void IntegerAndDecimalDivisionUseTheElementTypeRules()
    {
        CheckNonIeeeDivision<int>();
        CheckNonIeeeDivision<decimal>();
    }

    private static void CheckNonIeeeDivision<T>() where T : unmanaged, INumberBase<T>
    {
        T numerator = T.CreateChecked(-5), divisor = T.CreateChecked(2);
        T expected = numerator / divisor;
        var vector = new Vector<T>([numerator]);
        var matrix = Matrix<T>.FromColumnMajor(1, 1, [numerator]);
        var tensor = Tensor<T>.FromColumnMajor([1, 1, 1], [numerator]);

        Assert.Equal(expected, (vector / divisor)[0]);
        Assert.Equal(expected, (matrix / divisor)[0, 0]);
        Assert.Equal(expected, (tensor / divisor)[0, 0, 0]);
        Assert.Throws<DivideByZeroException>(() => vector / T.Zero);
        Assert.Throws<DivideByZeroException>(() => matrix / T.Zero);
        Assert.Throws<DivideByZeroException>(() => { tensor /= T.Zero; });
        Assert.Throws<DivideByZeroException>(() => XMath.Divide(tensor, new Tensor<T>(1, 1, 1)));
    }

    [Theory]
    [InlineData(0, 2, 3)]
    [InlineData(2, 0, 3)]
    [InlineData(2, 3, 0)]
    [InlineData(0, 2, int.MaxValue)]
    public void EmptyTensorsAndViewsKeepTheirShape(int rows, int columns, int count)
    {
        var tensor = new Tensor<int>(rows, columns, count);
        var vector = new Vector<int>(0);
        var matrix = new Matrix<int>(rows, 0);
        tensor /= 0;
        vector /= 0;
        matrix /= 0;

        Assert.Equal(0, (vector / 0).Length);
        Assert.Equal(0, (vector.AsVectorSpan().AsReadOnly() / 0).Length);
        var transposed = matrix.T / 0;
        Assert.Equal((0, rows), (transposed.Rows, transposed.Columns));
        var emptyMatrixView = new ReadOnlyMatrixSpan<int>([], [0, count], [0, 2]);
        var dividedMatrixView = emptyMatrixView / 0;
        Assert.Equal((0, count), (dividedMatrixView.Rows, dividedMatrixView.Columns));
        Assert.Equal(tensor.Lengths, (tensor / 0).Lengths);
        Assert.Equal(tensor.T.Lengths, (tensor.T.AsReadOnly() / 0).Lengths);
        Assert.Equal(tensor.Lengths, XMath.Divide(tensor, tensor).Lengths);
    }

    private static void AssertBits<T>(ReadOnlySpan<T> expected, ReadOnlySpan<T> actual) where T : unmanaged, IFloatingPointIeee754<T>
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            bool matches = T.IsNaN(expected[i])
                ? T.IsNaN(actual[i])
                : MemoryMarshal.AsBytes(expected.Slice(i, 1)).SequenceEqual(MemoryMarshal.AsBytes(actual.Slice(i, 1)));
            Assert.True(matches, $"The result at index {i} was {actual[i]}. Expected {expected[i]}.");
        }
    }
}
