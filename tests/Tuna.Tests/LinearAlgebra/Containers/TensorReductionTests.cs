namespace Tuna.Tests.LinearAlgebra;

using Tuna.LinearAlgebra;

[Collection(nameof(ParallelismCollection))]
[Trait("LinearAlgebra", "Batch")]
public class TensorReductionTests
{
    [Fact]
    public void SumMatricesMatchesVectorSumsOnDenseAndStridedViews()
    {
        var tensor = new Tensor<double>(5, 7, 132);
        for (int k = 0; k < tensor.Matrices.Count; k++)
        {
            for (int j = 0; j < 7; j++)
            {
                for (int i = 0; i < 5; i++)
                {
                    tensor[i, j, k] = (k % 3) switch
                    {
                        0 => 1e16,
                        1 => i + j * 5 + k,
                        _ => -1e16,
                    };
                }
            }
        }

        var dense = XMath.SumMatrices(tensor);
        AssertMatchesVectorSum(tensor, dense);

        var view = tensor[1..4, 2..6, 1..].T;
        var storage = new Matrix<double>(5, 6);
        storage.AsMatrixSpan().Fill(-100.0);
        var destination = storage[1..4, 1..5].T;
        XMath.SumMatrices(view, destination);
        AssertMatchesVectorSum(view, destination);
        Assert.Equal(-100.0, storage[0, 2]);
        Assert.Equal(-100.0, storage[3, 5]);
    }

    [Fact]
    public void SumMatricesHandlesBroadcastEmptyAndSingletonBatches()
    {
        var matrix = new Matrix<float>(1, 3);
        matrix[0, 0] = 2.5f;
        matrix[0, 1] = -0.0f;
        matrix[0, 2] = float.NaN;
        var repeated = XMath.SumMatrices(matrix.Broadcast(130));
        Assert.Equal(325f, repeated[0, 0]);
        Assert.Equal(0, BitConverter.SingleToInt32Bits(repeated[0, 1]));
        Assert.True(float.IsNaN(repeated[0, 2]));

        var once = XMath.SumMatrices(matrix.Broadcast(1));
        Assert.Equal(2.5f, once[0, 0]);
        Assert.Equal(0, BitConverter.SingleToInt32Bits(once[0, 1]));
        Assert.True(float.IsNaN(once[0, 2]));

        XMath.SumMatrices(new Tensor<float>(1, 3, 0), repeated);
        Assert.Equal(new float[3], repeated.AsSpan().ToArray());
        var empty = XMath.SumMatrices(new Tensor<int>(0, 3, 5));
        Assert.Equal((0, 3), (empty.Rows, empty.Columns));
    }

    [Fact]
    public void SumMatricesRejectsOverlapWithAnyInputMatrixAndWrongShape()
    {
        var tensor = new Tensor<double>(2, 3, 3);
        tensor.AsSpan().Fill(2.0);
        Assert.Throws<ArgumentException>("destination", () => XMath.SumMatrices(tensor, tensor.GetMatrix(2)));
        Assert.Throws<ArgumentException>("destination", () => XMath.SumMatrices(tensor, new Matrix<double>(3, 2)));
        Assert.All(tensor.AsSpan().ToArray(), value => Assert.Equal(2.0, value));
    }

    [Fact]
    public void SumMatricesIsDeterministicAndReusesStorage()
    {
        var input = new Tensor<double>(96, 48, 129);
        for (int i = 0; i < input.AsSpan().Length; i++)
        {
            input.AsSpan()[i] = (i % 7 - 3) * (i % 2 == 0 ? 1e16 : 1.0);
        }

        var destination = new Matrix<double>(96, 48);
        int previous = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = 1;
            var reference = XMath.SumMatrices(input);
            AssertMatchesVectorSum(input, reference);

            foreach (int degree in new[] { 1, previous })
            {
                LinearAlgebraSettings.MaxDegreeOfParallelism = degree;
                for (int i = 0; i < 20; i++)
                {
                    XMath.SumMatrices(input, destination);
                }

                GC.Collect(0, GCCollectionMode.Forced, blocking: true);
                long before = GC.GetAllocatedBytesForCurrentThread();
                for (int i = 0; i < 12; i++)
                {
                    XMath.SumMatrices(input, destination);
                }

                Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
                for (int i = 0; i < reference.AsSpan().Length; i++)
                {
                    Assert.Equal(BitConverter.DoubleToInt64Bits(reference.AsSpan()[i]), BitConverter.DoubleToInt64Bits(destination.AsSpan()[i]));
                }
            }
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = previous;
        }
    }

    private static void AssertMatchesVectorSum(ReadOnlyTensorSpan<double> x, ReadOnlyMatrixSpan<double> actual)
    {
        var values = new double[x.Matrices.Count];
        for (int j = 0; j < actual.Columns; j++)
        {
            for (int i = 0; i < actual.Rows; i++)
            {
                for (int k = 0; k < values.Length; k++)
                {
                    values[k] = x[i, j, k];
                }

                double expected = XMath.Sum(new ReadOnlyVectorSpan<double>(values));
                Assert.Equal(BitConverter.DoubleToInt64Bits(expected), BitConverter.DoubleToInt64Bits(actual[i, j]));
            }
        }
    }
}
