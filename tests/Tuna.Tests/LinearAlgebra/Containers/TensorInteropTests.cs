using Tuna.LinearAlgebra;
// Tuna.LinearAlgebra にも同名の TensorSpan・ReadOnlyTensorSpan があるので、名前空間の別名で区別する
using SystemTensors = System.Numerics.Tensors;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>
/// ライブラリ本体は System.Numerics.Tensors に依存しない。ビューの Data・Lengths・Strides を渡すだけで
/// TensorSpan と相互に変換できることを、テストだけがパッケージを参照して確かめる。
/// </summary>
[Trait("LinearAlgebra", "Interop")]
public class TensorInteropTests
{
    [Fact]
    public void ViewsConvertToTensorSpansWithoutCopying()
    {
        var a = Matrix<double>.FromRowMajor(3, 4, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        MatrixSpan<double> view = a;
        var tensor = new SystemTensors.TensorSpan<double>(view.Data, view.Lengths, view.Strides);
        Assert.Equal(7.0, tensor[[1, 2]]);
        tensor[[2, 3]] = 100;
        Assert.Equal(100.0, a[2, 3]);

        // 転置・部分行列・対角・n×1 の形（TensorSpan は長さ 1 の次元にストライド 0 を要求する）も同じ1行で渡せる
        var transposed = a.T;
        var t = new SystemTensors.ReadOnlyTensorSpan<double>(transposed.Data, transposed.Lengths, transposed.Strides);
        Assert.Equal(a[1, 2], t[[2, 1]]);
        var block = a[1.., 1..3];
        var b = new SystemTensors.ReadOnlyTensorSpan<double>(block.Data, block.Lengths, block.Strides);
        Assert.Equal(a[2, 2], b[[1, 1]]);
        var column = a[.., 1..2];
        var c = new SystemTensors.ReadOnlyTensorSpan<double>(column.Data, column.Lengths, column.Strides);
        Assert.Equal(a[2, 1], c[[2, 0]]);
        var diagonal = a.Diagonal();
        var d = new SystemTensors.ReadOnlyTensorSpan<double>(diagonal.Data, diagonal.Lengths, diagonal.Strides);
        Assert.Equal(a[2, 2], d[[2]]);
    }

    [Fact]
    public void RowMajorTensorsBecomeStridedViews()
    {
        double[] values = [1, 2, 3, 4, 5, 6];
        var tensor = new SystemTensors.TensorSpan<double>(values, [2, 3], []);
        var view = MatrixSpan<double>.Create(ref tensor.GetPinnableReference(), tensor.Lengths, tensor.Strides);
        Assert.Equal(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } }, view.ToArray());
        var x = new Vector<double>([1, 1, 1]);
        Assert.Equal(new double[] { 6, 15 }, XMath.Multiply(view, x).ToArray());
    }

    [Theory]
    [InlineData(0, 5)]
    [InlineData(5, 0)]
    [InlineData(0, 0)]
    public void EmptyMatricesRoundTripThroughTheirLayout(int rows, int columns)
    {
        // 要素のない配置は、長さ 0 でない次元のストライドが 0 でも書き込み可能なビューとして作り直せる
        MatrixSpan<double> view = new Matrix<double>(rows, columns);
        var again = new MatrixSpan<double>(view.Data, view.Lengths, view.Strides);
        Assert.Equal((rows, columns), (again.Rows, again.Columns));
        var tensor = new SystemTensors.TensorSpan<double>(view.Data, view.Lengths, view.Strides);
        var fromTensor = MatrixSpan<double>.Create(ref tensor.GetPinnableReference(), tensor.Lengths, tensor.Strides);
        Assert.Equal((rows, columns), (fromTensor.Rows, fromTensor.Columns));
    }
}
