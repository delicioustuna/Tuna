using Tuna.LinearAlgebra;

namespace Tuna.Tests.LinearAlgebra;

public class TraceTests
{
    [Fact]
    public void TraceReadsOnlyTheMainDiagonalOfRectangularViews()
    {
        double[] storage = [4, double.NaN, double.NaN, double.NaN, 5, double.NaN, double.NaN, double.NaN];
        var matrix = new MatrixSpan<double>(storage, [2, 3], [1, 3]);

        Assert.Equal(9, XMath.Trace(matrix));
        Assert.Equal(9, XMath.Trace(matrix.T.AsReadOnly()));
        Assert.Equal(5, XMath.Trace(matrix[1.., 1..]));
        Assert.Equal(13, XMath.Trace(new Vector<double>([4, 9, 7]).BroadcastRows(2)));
        Assert.Equal(9, XMath.Trace(new Matrix<int>(new int[,] { { 4, 0 }, { 0, 5 } })));
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(0, 3)]
    [InlineData(3, 0)]
    public void TraceOfAnEmptyMatrixIsZero(int rows, int columns)
    {
        Assert.Equal(0, XMath.Trace(new Matrix<double>(rows, columns)));
    }

    [Fact]
    public void TraceUsesSumSemanticsForNonFiniteDiagonalEntries()
    {
        var matrix = new Matrix<float>(2, 2);
        matrix[0, 0] = float.PositiveInfinity;
        Assert.Equal(float.PositiveInfinity, XMath.Trace(matrix));

        matrix[1, 1] = float.NegativeInfinity;
        Assert.True(float.IsNaN(XMath.Trace(matrix)));

        matrix[0, 0] = float.NaN;
        matrix[1, 1] = 0;
        Assert.True(float.IsNaN(XMath.Trace(matrix)));
    }
}
