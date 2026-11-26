using System.Numerics;
using System.Runtime.CompilerServices;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Tests.Differentiation;

[Collection(nameof(ParallelismCollection))]
public class TensorNodeTests
{
    [Fact]
    public void FloatReferencesMatchNumPy() => CheckReferences<float>();

    [Fact]
    public void DoubleReferencesMatchNumPy() => CheckReferences<double>();

    [Fact]
    public void RightOperandBroadcastUsesTheReductionAfterBatchedMultiplication()
    {
        var tape = new Tape();
        var left = new Tensor<double>(2, 3, 5);
        var right = new Matrix<double>(4, 3);
        for (int i = 0; i < left.AsSpan().Length; i++)
        {
            left.AsSpan()[i] = (i - 11) * 0.125;
        }

        right.AsSpan().Fill(0.25);
        var leftNode = tape.Constant(left[Range.All, Range.All, 1..4].AsReadOnly());
        var weights = tape.Leaf(right);
        var y = XMath.Multiply(leftNode, weights.T.Broadcast(3));
        var expected = XMath.Multiply(left[Range.All, Range.All, 1..4], right.T.Broadcast(3));
        Assert.Equal(expected.AsSpan().ToArray(), y.Value.Data.ToArray());
        var dy = new Matrix<double>(2, 4);
        dy.AsSpan().Fill(1);
        var partial = XMath.Multiply(left[Range.All, Range.All, 1..4].T, dy.AsMatrixSpan().Broadcast(3));
        var gradient = XMath.SumMatrices(partial);
        tape.Backward(WeightedSum(tape, XMath.SumMatrices(y), dy));
        Assert.Equal(gradient.T.ToMatrix().AsSpan().ToArray(), tape.GetGradient(weights).Data.ToArray());
    }

    [Fact]
    public void TransposedTensorLeavesUseLogicalOrderInTheFlatGradient()
    {
        var view = new Tensor<double>(2, 3, 5).T;
        var tape = new Tape();
        var node = tape.Leaf(view);
        var selected = node[1.., .., 1..4];
        var weights = new Matrix<double>(2, 2);
        double[] coefficients = [1, 2, 3, 4];
        coefficients.CopyTo(weights.AsSpan());
        var loss = WeightedSum(tape, XMath.SumMatrices(selected), weights);
        var flat = new double[30];
        tape.Gradient(loss, flat);
        var gradient = tape.GetGradient(node);

        for (int k = 0; k < 5; k++)
        {
            for (int j = 0; j < 2; j++)
            {
                for (int i = 0; i < 3; i++)
                {
                    double expected = k >= 1 && k < 4 && i > 0 ? weights[i - 1, j] : 0;
                    Assert.Equal(expected, gradient[i, j, k]);
                    Assert.Equal(expected, flat[(k * 2 + j) * 3 + i]);
                }
            }
        }
    }

    [Fact]
    public void MatrixExtractionAndPackingAccumulateIntoTheSameTensor()
    {
        var tensor = new Tensor<double>(2, 3, 4);
        for (int i = 0; i < tensor.AsSpan().Length; i++)
        {
            tensor.AsSpan()[i] = i;
        }

        var tape = new Tape();
        var node = tape.Leaf(tensor);
        var view = node.T[1.., .., 1..3];
        var first = view.GetMatrix(0);
        var packed = view.ToMatrix();
        Assert.Equal(new double[] { 8, 10, 9, 11 }, first.Value.ToMatrix().AsSpan().ToArray());
        Assert.Equal(new double[] { 8, 10, 9, 11, 14, 16, 15, 17 }, packed.Value.Data.ToArray());

        var loss = XMath.Sum(XMath.ColumnSums(first)) + XMath.Sum(XMath.ColumnSums(packed));
        var destination = new Tensor<double>(2, 3, 4);
        tape.Gradient(loss, node, destination.AsTensorSpan());
        for (int k = 0; k < 4; k++)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int i = 0; i < 2; i++)
                {
                    double expected = j == 0 ? 0 : k == 1 ? 2 : k == 2 ? 1 : 0;
                    Assert.Equal(expected, destination[i, j, k]);
                }
            }
        }
    }

    [Fact]
    public void BroadcastReductionAddsToAStridedParameterView()
    {
        var tape = new Tape();
        var parameter = Parameter<double>.Matrix(3, 4);
        parameter.Gradient.Fill(2);
        var source = tape.UseMatrix(parameter).T[1..3, ..2];
        var view = source.Broadcast(8).T[1..2, .., 2..5];
        tape.Backward(XMath.Sum(XMath.ColumnSums(XMath.SumMatrices(view))));

        for (int j = 0; j < 4; j++)
        {
            for (int i = 0; i < 3; i++)
            {
                Assert.Equal(i == 1 && (j == 1 || j == 2) ? 5 : 2, parameter.Gradient[i + j * 3]);
            }
        }

        Assert.Equal(5d, tape.GetGradient(view)[0, 1, 2]);
    }

    [Fact]
    public void TensorParametersAndConstantsPreserveStorage()
    {
        var parameter = Parameter<float>.Tensor(2, 3, 4);
        var tape = new Tape();
        var tensor = new Tensor<float>(2, 3, 4);
        var node = tape.Constant(tensor);
        Assert.True(Unsafe.AreSame(ref tensor.AsSpan()[0], ref Unsafe.AsRef(in node.Value[0, 0, 0])));
        var constant = tape.ConstantTensor(parameter);
        var learnable = tape.UseTensor(parameter);
        Assert.False(constant.RequiresGradient);
        Assert.True(learnable.RequiresGradient);
        Assert.True(Unsafe.AreSame(ref parameter.Value[0], ref Unsafe.AsRef(in constant.Value[0, 0, 0])));
        Assert.Throws<ArgumentException>(() => tape.UseTensor(Parameter<float>.Matrix(2, 3)));

        var loss = XMath.Sum(XMath.ColumnSums(XMath.SumMatrices(learnable)));
        tape.Backward(loss);
        Assert.All(parameter.Gradient.ToArray(), x => Assert.Equal(1f, x));
    }

    [Fact]
    public void EmptyBatchesAndZeroRowsPreserveShape()
    {
        var tape = new Tape();
        var matrix = tape.Leaf(new Matrix<double>(3, 2));
        var right = tape.Leaf(new Tensor<double>(2, 4, 0));
        var result = matrix.Broadcast(0) * right;
        Assert.Equal((3, 4, 0), (result.Rows, result.Columns, result.MatrixCount));
        var summed = XMath.SumMatrices(result);
        var empty = tape.Leaf(new Tensor<double>(0, int.MaxValue, 2));
        Assert.Equal((0, int.MaxValue), (empty.GetMatrix(1).Rows, empty.GetMatrix(1).Columns));
        Assert.Throws<OverflowException>(() => empty.ToMatrix());
        tape.Backward(XMath.Sum(XMath.ColumnSums(summed)));
        Assert.All(tape.GetGradient(matrix).Data.ToArray(), x => Assert.Equal(0d, x));
        Assert.True(tape.GetGradient(empty).IsEmpty);
    }

    [Fact]
    public void InvalidInputsDoNotChangeTheRecording()
    {
        var tape = new Tape();
        var other = new Tape();
        var input = tape.Leaf(new Tensor<double>(2, 3, 4));
        var foreign = other.Leaf(new Tensor<double>(3, 5, 4));
        var foreignWrongShape = other.Leaf(new Tensor<double>(7, 5, 2));
        Assert.Throws<InvalidOperationException>(() => input * foreign);
        Assert.Throws<InvalidOperationException>(() => input * foreignWrongShape);
        Assert.Throws<InvalidOperationException>(() => default(TensorNode<double>) * input);
        var table = tape.Leaf(new Matrix<double>(2, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => XMath.Embedding(table, [1, -1]));
        Assert.Throws<ArgumentOutOfRangeException>(() => XMath.Embedding(table, [4]));
        Assert.Throws<ArgumentOutOfRangeException>(() => input.GetMatrix(4));

        // 例外後も Reset せずに同じテープで正常に記録・逆伝播できる
        var right = tape.Leaf(new Tensor<double>(3, 2, 4));
        var validProduct = input * right;
        tape.Backward(XMath.Sum(XMath.ColumnSums(XMath.SumMatrices(validProduct))));
        var inputGrad = tape.GetGradient(input);
        Assert.Equal((2, 3, 4), (inputGrad.Lengths[0], inputGrad.Lengths[1], inputGrad.Lengths[2]));

        tape.Reset();
        Assert.Throws<InvalidOperationException>(() => input.GetMatrix(0));
        Assert.Throws<InvalidOperationException>(() => _ = input.Value);
        Assert.Throws<InvalidOperationException>(() => input.ToMatrix());
    }

    [Fact]
    public void NonBroadcastSelfProductAndDistinctSliceAdjointsMatchPerMatrixProducts()
    {
        var tape = new Tape();
        var aData = new Tensor<double>(3, 3, 4);
        var bData = new Tensor<double>(3, 2, 4);
        var seedData = new Matrix<double>(3, 8);
        var seedSelfData = new Matrix<double>(3, 12);
        for (int i = 0; i < aData.AsSpan().Length; i++)
        {
            aData.AsSpan()[i] = ((i % 7) - 3) * 0.25;
        }

        for (int i = 0; i < bData.AsSpan().Length; i++)
        {
            bData.AsSpan()[i] = ((i % 5) - 2) * 0.3;
        }

        for (int i = 0; i < seedData.AsSpan().Length; i++)
        {
            seedData.AsSpan()[i] = ((i % 9) - 4) * 0.2;
        }

        for (int i = 0; i < seedSelfData.AsSpan().Length; i++)
        {
            seedSelfData.AsSpan()[i] = ((i % 11) - 5) * 0.15;
        }

        // 非ブロードキャスト Tensor × Tensor（スライスごとに異なる随伴）
        var a = tape.Leaf(aData);
        var b = tape.Leaf(bData);
        var y = a * b;
        tape.Backward(WeightedSum(tape, y.ToMatrix(), seedData));

        var refTape = new Tape();
        var refA = new MatrixNode<double>[4];
        var refB = new MatrixNode<double>[4];
        ScalarNode refLoss = 0;
        for (int k = 0; k < 4; k++)
        {
            refA[k] = refTape.Leaf(aData.GetMatrix(k).AsReadOnly());
            refB[k] = refTape.Leaf(bData.GetMatrix(k).AsReadOnly());
            var sliceSeed = seedData[.., (k * 2)..((k + 1) * 2)].ToMatrix();
            refLoss += WeightedSum(refTape, refA[k] * refB[k], sliceSeed);
        }

        refTape.Backward(refLoss);
        var gradA = tape.GetGradient(a);
        var gradB = tape.GetGradient(b);
        for (int k = 0; k < 4; k++)
        {
            Assert.Equal(refTape.GetGradient(refA[k]).ToMatrix().AsSpan().ToArray(), gradA.GetMatrix(k).ToMatrix().AsSpan().ToArray());
            Assert.Equal(refTape.GetGradient(refB[k]).ToMatrix().AsSpan().ToArray(), gradB.GetMatrix(k).ToMatrix().AsSpan().ToArray());
        }

        // 自己積 a * a（初回上書き + 2回目加算の両経路を通る）
        tape.Reset();
        var selfA = tape.Leaf(aData);
        var selfY = selfA * selfA;
        tape.Backward(WeightedSum(tape, selfY.ToMatrix(), seedSelfData));

        refTape.Reset();
        ScalarNode selfRefLoss = 0;
        for (int k = 0; k < 4; k++)
        {
            refA[k] = refTape.Leaf(aData.GetMatrix(k).AsReadOnly());
            var sliceSeed = seedSelfData[.., (k * 3)..((k + 1) * 3)].ToMatrix();
            selfRefLoss += WeightedSum(refTape, refA[k] * refA[k], sliceSeed);
        }

        refTape.Backward(selfRefLoss);
        var gradSelf = tape.GetGradient(selfA);
        for (int k = 0; k < 4; k++)
        {
            var expected = refTape.GetGradient(refA[k]).ToMatrix().AsSpan().ToArray();
            var actual = gradSelf.GetMatrix(k).ToMatrix().AsSpan().ToArray();
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], actual[i], 13);
            }
        }
    }

    [Fact]
    public void LargeTensorExceedingSharedArrayPoolLimitReusesTapeScratchWithZeroAllocations()
    {
        // 128 × 128 × 65 = 1,064,960 要素（ArrayPool<T>.Shared の 2^20 上限超）
        var tape = new Tape();
        var left = new Tensor<float>(128, 128, 65);
        var right = new Tensor<float>(128, 1, 65);
        left.AsSpan().Fill(0.125f);
        right.AsSpan().Fill(0.25f);

        void Run()
        {
            tape.Reset();
            var a = tape.Leaf(left);
            var b = tape.Leaf(right);
            var y = a * b;
            tape.Backward(XMath.Sum(XMath.ColumnSums(y.ToMatrix())));
        }

        for (int i = 0; i < 3; i++)
        {
            Run();
        }

        long bytes = -1;
        for (int trial = 0; trial < 3 && bytes != 0; trial++)
        {
            GC.Collect(0, GCCollectionMode.Forced, blocking: true);
            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 3; i++)
            {
                Run();
            }

            bytes = GC.GetAllocatedBytesForCurrentThread() - before;
        }

        Assert.Equal(0, bytes);
    }

    [Fact]
    public void ChunkedBroadcastProductBackwardMatchesSumOfMatrixProducts()
    {
        // 512 × 512 × 20 = 5,242,880 要素 > MaxBroadcastScratchElements (4,194,304) でチャンク分割経路を検証する
        const int rows = 512;
        const int cols = 512;
        const int count = 20;
        var aData = new Matrix<float>(rows, cols);
        var bData = new Tensor<float>(cols, 1, count);
        aData.AsSpan().Fill(0.01f);
        for (int k = 0; k < count; k++)
        {
            bData.GetMatrix(k).Fill(k + 1);
        }

        var tape = new Tape();
        var a = tape.Leaf(aData);
        var b = tape.Constant(bData);
        var y = a.Broadcast(count) * b;
        tape.Backward(XMath.Sum(XMath.ColumnSums(y.ToMatrix())));

        // dy は全要素 1、各バッチ k の寄与は 1 * (k+1) = k+1。合計は 20 * 21 / 2 = 210
        var gradA = tape.GetGradient(a);
        Assert.Equal(210f, gradA[0, 0]);
        Assert.Equal(210f, gradA[rows - 1, cols - 1]);
    }

    [Fact]
    public void ConstantsUnreachedBranchesAndEmptyEmbeddingWorkAsSpecified()
    {
        var tape = new Tape();
        var a = tape.Constant(new Matrix<double>(2, 2));
        var b = tape.Constant(new Tensor<double>(2, 2, 3));
        var c = a.Broadcast(3) * b;
        var s = XMath.SumMatrices(c);
        var emb = XMath.Embedding(s, []);
        Assert.False(c.RequiresGradient);
        Assert.False(s.RequiresGradient);
        Assert.False(emb.RequiresGradient);
        Assert.Equal((2, 0), (emb.Rows, emb.Columns));

        // 未到達枝の Infinity が到達枝の勾配を汚染しない
        var leaf = tape.Leaf(Matrix<double>.FromColumnMajor(2, 2, [1, 2, 3, 4]));
        var inf = new Tensor<double>(2, 2, 2);
        inf.AsSpan().Fill(double.PositiveInfinity);
        _ = leaf.Broadcast(2) * tape.Constant(inf);
        var used = XMath.Embedding(leaf, [0, 1, 0]);
        tape.Backward(XMath.Sum(XMath.ColumnSums(used)));
        var grad = tape.GetGradient(leaf);
        Assert.Equal(2.0, grad[0, 0]);
        Assert.Equal(2.0, grad[1, 0]);
        Assert.Equal(1.0, grad[0, 1]);
        Assert.Equal(1.0, grad[1, 1]);

        // 逆伝播完了後も全定数の演算は評価できる
        var afterBackward = a.Broadcast(3) * b;
        Assert.False(afterBackward.RequiresGradient);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void TensorAndEmbeddingRecordingReusesItsBuffers(int degree)
    {
        int previous = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = degree;
            var tape = new Tape();
            var left = new Matrix<double>(17, 33);
            var right = new Tensor<double>(33, 37, 32);
            left.AsSpan().Fill(0.25);
            right.AsSpan().Fill(-0.5);
            int[] ids = [0, 1, 0, 7, 15, 36, 27, 12];
            for (int i = 0; i < 4; i++)
            {
                TrainingStep(tape, left, right, ids);
            }

            long bytes = -1;
            for (int trial = 0; trial < 3 && bytes != 0; trial++)
            {
                GC.Collect(0, GCCollectionMode.Forced, blocking: true);
                long before = GC.GetAllocatedBytesForCurrentThread();
                for (int i = 0; i < 6; i++)
                {
                    TrainingStep(tape, left, right, ids);
                }

                bytes = GC.GetAllocatedBytesForCurrentThread() - before;
            }

            Assert.Equal(0, bytes);
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = previous;
        }
    }

    private static void TrainingStep(Tape tape, Matrix<double> left, Tensor<double> right, int[] ids)
    {
        tape.Reset();
        var product = tape.Leaf(left).Broadcast(32) * tape.Leaf(right);
        var embedding = XMath.Embedding(XMath.SumMatrices(product), ids);
        tape.Backward(XMath.Sum(XMath.ColumnSums(embedding)));
    }

    private static void CheckReferences<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        var left = MatrixFrom<T>(3, 2, TensorReferences.Left);
        var right = new Tensor<T>(2, 4, 3);
        Copy<T>(TensorReferences.Right, right.AsSpan());
        var tape = new Tape();
        var a = tape.Leaf(left);
        var b = tape.Leaf(right.T);
        var product = a.Broadcast(3) * b.T;
        Assert.Equal(Convert<T>(TensorReferences.Product), product.Value.ToTensor().AsSpan().ToArray());
        var seed = MatrixFrom<T>(3, 4, TensorReferences.ProductSeed);
        var loss = WeightedSum(tape, XMath.SumMatrices(product), seed);
        tape.Backward(loss);
        Assert.Equal(Convert<T>(TensorReferences.LeftGradient), tape.GetGradient(a).Data.ToArray());
        Assert.Equal(Convert<T>(TensorReferences.RightGradient), tape.GetGradient(b).T.ToTensor().AsSpan().ToArray());

        T original = left[0, 0];
        T h = T.CreateChecked(0.125);
        left[0, 0] = original + h;
        double plus = RawProductLoss(left, right, seed);
        left[0, 0] = original - h;
        double minus = RawProductLoss(left, right, seed);
        Assert.Equal(TensorReferences.LeftGradient[0], (plus - minus) / 0.25, 8);

        tape.Reset();
        var weights = Parameter<T>.Matrix(3, 4);
        Copy<T>(TensorReferences.Table, weights.Value);
        var table = tape.UseMatrix(weights);
        int[] ids = (int[])TensorReferences.Indices.Clone();
        var result = XMath.Embedding(table, ids);
        Assert.Equal(Convert<T>(TensorReferences.Embedding), result.Value.Data.ToArray());
        Array.Fill(ids, 1);
        tape.Backward(WeightedSum(tape, result, MatrixFrom<T>(3, ids.Length, TensorReferences.EmbeddingSeed)));
        Assert.Equal(Convert<T>(TensorReferences.TableGradient), weights.Gradient.ToArray());
    }

    private static double RawProductLoss<T>(Matrix<T> a, Tensor<T> b, Matrix<T> seed) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var sums = XMath.SumMatrices(XMath.Multiply(a.AsMatrixSpan().Broadcast(3), b));
        return double.CreateChecked(XMath.Sum(XMath.ColumnSums(XMath.Hadamard(sums, seed))));
    }

    private static ScalarNode WeightedSum<T>(Tape tape, MatrixNode<T> node, Matrix<T> weights)
        where T : unmanaged, IFloatingPointIeee754<T>
        => XMath.Sum(XMath.ColumnSums(XMath.Hadamard(node, tape.Constant(weights))));

    private static Matrix<T> MatrixFrom<T>(int rows, int columns, double[] values) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var matrix = new Matrix<T>(rows, columns);
        Copy<T>(values, matrix.AsSpan());
        return matrix;
    }

    private static T[] Convert<T>(double[] values) where T : unmanaged, IFloatingPointIeee754<T> =>
        Array.ConvertAll(values, value => T.CreateChecked(value));

    private static void Copy<T>(double[] values, Span<T> destination) where T : unmanaged, IFloatingPointIeee754<T>
    {
        for (int i = 0; i < values.Length; i++)
        {
            destination[i] = T.CreateChecked(values[i]);
        }
    }
}
