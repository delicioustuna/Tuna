using Tuna.LinearAlgebra;

namespace Tuna.Tests.LinearAlgebra;

public class IndexedGenerationTests
{
    private readonly struct VectorPosition : IVectorIndexOperator<double>
    {
        public static double Invoke(int index) => index;
    }

    private readonly struct VectorShift(double offset) : IStatefulVectorIndexOperator<double>
    {
        public double Invoke(int index) => index + offset;
    }

    private readonly struct TensorPosition : ITensorIndexOperator<double>
    {
        public static double Invoke(ReadOnlySpan<int> indices) => indices[0] + 10 * indices[1] + 100 * indices[2];
    }

    private readonly struct TensorShift(double offset) : IStatefulTensorIndexOperator<double>
    {
        public double Invoke(ReadOnlySpan<int> indices) => TensorPosition.Invoke(indices) + offset;
    }

    [Fact]
    public void VectorGenerateSupportsOperatorsLambdasAndStridedDestinations()
    {
        double step = 0.25;
        var vector = Vector<double>.Generate(3, i => i * step);
        Assert.Equal(new double[] { 0, 0.25, 0.5 }, vector.ToArray());
        Assert.Equal(new double[] { 0, 1, 2 }, Vector<double>.Generate<VectorPosition>(3).ToArray());
        Assert.Equal(new double[] { 2, 3, 4 }, Vector<double>.Generate(3, new VectorShift(2)).ToArray());

        Assert.Same(vector, XMath.Generate(new VectorShift(2), vector));
        Assert.Equal(new double[] { 2, 3, 4 }, vector.ToArray());
        XMath.Generate<VectorPosition>(vector);
        Assert.Equal(new double[] { 0, 1, 2 }, vector.ToArray());

        var matrix = new Matrix<double>(2, 3);
        XMath.Generate(static i => 10 + i, matrix.Row(1));
        Assert.Equal(new double[,] { { 0, 0, 0 }, { 10, 11, 12 } }, matrix.ToArray());

        var bytes = new Vector<byte>(2);
        Assert.Same(bytes, XMath.Generate<byte>(static _ => 7, bytes));
        Assert.Equal(new byte[] { 7, 7 }, bytes.ToArray());
    }

    [Fact]
    public void TensorGeneratePassesCoordinatesInShapeOrder()
    {
        ReadOnlySpan<nint> shape = [2, 2, 2];
        var tensor = Tensor<double>.Generate(shape, TensorPosition.Invoke);
        Assert.Equal(new double[] { 0, 1, 10, 11, 100, 101, 110, 111 }, tensor.AsSpan().ToArray());
        Assert.Equal(tensor.AsSpan().ToArray(), Tensor<double>.Generate<TensorPosition>(shape).AsSpan().ToArray());
        Assert.Equal(113, Tensor<double>.Generate(shape, new TensorShift(2))[1, 1, 1]);

        Assert.Same(tensor, XMath.Generate(new TensorShift(2), tensor));
        Assert.Equal(113, tensor[1, 1, 1]);
        XMath.Generate<TensorPosition>(tensor);
        Assert.Equal(111, tensor[1, 1, 1]);
        Assert.Same(tensor, XMath.Generate(static indices => TensorPosition.Invoke(indices) + 1, tensor));
        Assert.Equal(112, tensor[1, 1, 1]);

        var storage = new Tensor<double>(4, 3, 4);
        var view = storage[1..3, .., 1..3].T;
        int calls = 0;
        XMath.Generate(indices =>
        {
            Assert.Equal(3, indices.Length);
            calls++;
            return TensorPosition.Invoke(indices) + 1;
        }, view);
        Assert.Equal(12, calls);
        Assert.Equal(113, storage[2, 2, 2]);
        Assert.Equal(1, storage[1, 0, 1]);
        Assert.Equal(0, storage[0, 0, 1]);
        Assert.Equal(0, storage[1, 0, 0]);
    }

    [Fact]
    public void MatrixGenerationLambdasUseTheExistingLayouts()
    {
        static double Entry(int i, int j) => 1 + 10 * i + j;

        var full = Matrix<double>.Generate(3, 3, Entry);
        Assert.Equal(new double[,] { { 1, 2, 3 }, { 11, 12, 13 }, { 21, 22, 23 } }, full.ToArray());
        var dest = new Matrix<double>(3, 3);
        XMath.Generate(static (i, j) => 1 + 10 * i + j, dest.T);
        Assert.Equal(full.ToArray(), dest.T.ToArray());

        var band = Matrix<double>.GenerateBand(3, 3, 1, 0, Entry);
        Assert.Equal(new double[,] { { 1, 0, 0 }, { 11, 12, 0 }, { 0, 22, 23 } }, band.ToArray());
        Assert.Same(dest, XMath.GenerateBand(1, 0, Entry, dest));
        Assert.Equal(band.ToArray(), dest.ToArray());

        var symmetric = Matrix<double>.GenerateSymmetric(3, Entry);
        Assert.Equal(new double[,] { { 1, 11, 21 }, { 11, 12, 22 }, { 21, 22, 23 } }, symmetric.ToArray());
        XMath.GenerateSymmetric(Entry, dest);
        Assert.Equal(symmetric.ToArray(), dest.ToArray());

        var symmetricBand = Matrix<double>.GenerateSymmetric(3, 1, Entry);
        Assert.Equal(new double[,] { { 1, 11, 0 }, { 11, 12, 22 }, { 0, 22, 23 } }, symmetricBand.ToArray());
        XMath.GenerateSymmetric(1, Entry, dest);
        Assert.Equal(symmetricBand.ToArray(), dest.ToArray());
    }

    [Fact]
    public void GenerationValidatesShapeAndDoesNotEvaluateEmptyOutputs()
    {
        Assert.Throws<ArgumentNullException>("op", () => Vector<double>.Generate(0, null!));
        Assert.Throws<ArgumentNullException>("op", () => Matrix<double>.Generate(0, 0, null!));
        Assert.Throws<ArgumentNullException>("op", () => Tensor<double>.Generate([0, 0, 0], null!));
        Assert.Throws<ArgumentOutOfRangeException>("length", () => Vector<double>.Generate<VectorPosition>(-1));
        Assert.Throws<ArgumentOutOfRangeException>("lengths", () => Tensor<double>.Generate<TensorPosition>([1, -1, 1]));
        Assert.Throws<NotSupportedException>(() => Tensor<double>.Generate([1, 1, 1, 1], TensorPosition.Invoke));
        Assert.Throws<OverflowException>(() => Tensor<double>.Generate<TensorPosition>([int.MaxValue, 2, 1]));

        Assert.Equal(0, Vector<double>.Generate(0, static _ => throw new InvalidOperationException()).Length);
        Assert.True(Tensor<double>.Generate([2, 0, 3], static _ => throw new InvalidOperationException()).IsEmpty);
    }
}
