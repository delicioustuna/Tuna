using System.Numerics;
using Tuna.LinearAlgebra;
using DoubleVector = Tuna.LinearAlgebra.Vector<double>;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>公開コンテナーの格納順、コピー・共有契約と既存演算への接続。</summary>
[Trait("LinearAlgebra", "Containers")]
public class DenseContainerTests
{
    [Fact]
    public void MatrixArrayConversionsCopyAndPreserveNonSquareLayout()
    {
        var source = new float[,] { { 1, 2, 3 }, { 4, 5, 6 } };
        var matrix = source.ToMatrix();
        Assert.Equal(2, matrix.Rows);
        Assert.Equal(3, matrix.Columns);
        Assert.False(matrix.IsSquare);
        Assert.Equal(new float[] { 1, 4, 2, 5, 3, 6 }, matrix.AsSpan().ToArray());
        Assert.Equal(source, matrix.ToArray());
        Assert.Equal(source, Matrix<float>.FromRowMajor(2, 3, [1, 2, 3, 4, 5, 6]).ToArray());
        Assert.Equal(source, Matrix<float>.FromColumnMajor(2, 3, [1, 4, 2, 5, 3, 6]).ToArray());
        var cast = (Matrix<float>)source;
        var output = (float[,])matrix;
        source[0, 0] = 99;
        output[1, 2] = 99;
        Assert.Equal(1f, matrix[0, 0]);
        Assert.Equal(6f, matrix[1, 2]);
        Assert.Equal(1f, cast[0, 0]);
        var clone = matrix.Clone();
        clone[0, 0] = 77;
        Assert.Equal(1f, matrix[0, 0]);
        matrix.Column(1)[0] = 42;
        Assert.Equal(42f, matrix[0, 1]);
    }

    [Fact]
    public void VectorArrayAndSpanConversionsAlwaysCopy()
    {
        double[] source = [1, 2, 3, 4, 5];
        var vector = source.ToVector();
        var fromSpan = source.AsSpan(1, 3).ToVector();
        ReadOnlySpan<double> readOnly = source.AsSpan(2, 2);
        var fromReadOnly = readOnly.ToVector();
        var cast = (DoubleVector)source;
        var output = (double[])vector;
        source[0] = 99;
        source[1] = 99;
        source[2] = 99;
        output[0] = 88;
        Assert.Equal(new double[] { 1, 2, 3, 4, 5 }, vector.ToArray());
        Assert.Equal(new double[] { 2, 3, 4 }, fromSpan.ToArray());
        Assert.Equal(new double[] { 3, 4 }, fromReadOnly.ToArray());
        Assert.Equal(1d, cast[0]);
        var clone = vector.Clone();
        clone[0] = 77;
        Assert.Equal(1d, vector[0]);
        vector.AsSpan()[0] = 42;
        Assert.Equal(42d, vector[0]);
    }

    [Fact]
    public void AsConversionsShareStorageInBothDirections()
    {
        var vector = Enumerable.Range(0, 24).ToArray().ToVector();
        var matrix = vector.AsMatrix(4, 6);
        matrix[3, 5] = 100;
        Assert.Equal(100, vector[23]);
        vector[0] = 200;
        Assert.Equal(200, matrix[0, 0]);
        matrix.AsVector()[7] = 500;
        Assert.Equal(500, vector[7]);
    }

    [Fact]
    public void ToConversionsAndCloneHaveIndependentStorage()
    {
        var vector = Enumerable.Range(0, 12).ToArray().ToVector();
        var matrix = vector.AsMatrix(3, 4);
        var copies = new[]
        {
            matrix.ToVector(), vector.Clone(),
            vector.ToMatrix(3, 4).AsVector(), matrix.Clone().AsVector()
        };
        vector[0] = 99;
        foreach (var copy in copies)
        {
            Assert.Equal(0, copy[0]);
            copy[1] = 88;
        }

        Assert.Equal(1, vector[1]);
    }

    [Fact]
    public void CheckedNumericConversionsPreserveShapeAndDoNotShare()
    {
        var matrix = new Matrix<int>(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var doubles = matrix.ConvertChecked<double>();
        Assert.Equal(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } }, doubles.ToArray());
        var singles = matrix.AsVector().ConvertChecked<float>();
        Assert.Equal(new float[] { 1, 4, 2, 5, 3, 6 }, singles.ToArray());
        matrix[0, 0] = 99;
        Assert.Equal(1d, doubles[0, 0]);
        Assert.Equal(1f, singles[0]);
        Assert.Throws<OverflowException>(() => new int[] { 256 }.ToVector().ConvertChecked<byte>());
        Assert.Throws<OverflowException>(() => new Matrix<int>(new int[,] { { -1 } }).ConvertChecked<uint>());
    }

    [Fact]
    public void ComplexValuesSurviveEveryLayoutConversion()
    {
        var values = new Complex[,] { { new(1, 2), new(3, -4) }, { new(-5, 6), new(7, 8) } };
        var matrix = values.ToMatrix();
        Assert.Equal(values, matrix.ToVector().ToMatrix(2, 2).ToArray());
        Assert.Equal(new Complex(3, -4), matrix.T[1, 0]);
        Assert.Equal(matrix.T.ToArray(), XMath.Transpose(matrix).ToArray());
        Assert.Equal(matrix.AsSpan().ToArray(), matrix.ConvertChecked<Complex>().AsSpan().ToArray());
    }

    [Fact]
    public void InvalidShapesLengthsAndIndicesAreRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Matrix<int>(-1, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Tuna.LinearAlgebra.Vector<int>(-1));
        Assert.Throws<OverflowException>(() => new Matrix<int>(int.MaxValue, 2));
        Assert.Throws<ArgumentException>(() => Matrix<int>.FromColumnMajor(2, 3, [1, 2]));
        Assert.Throws<ArgumentException>(() => Matrix<int>.FromRowMajor(2, 3, [1, 2]));
        var matrix = new Matrix<int>(2, 3);
        var vector = matrix.AsVector();
        Assert.Throws<ArgumentException>(() => vector.AsMatrix(3, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => matrix[-1, 0]);
        Assert.Throws<ArgumentOutOfRangeException>(() => matrix[0, 3] = 1);
        Assert.Throws<ArgumentOutOfRangeException>(() => matrix.Column(3).ToArray());
        Assert.Throws<ArgumentOutOfRangeException>(() => vector[6]);
        Assert.Throws<ArgumentOutOfRangeException>(() => vector[-1] = 1);
        Assert.Throws<ArgumentNullException>(() => new Matrix<int>((int[,])null!));
        Assert.Throws<ArgumentNullException>(() => ((int[])null!).ToVector());
    }

    [Fact]
    public void DoubleOperationsUseExistingBlasResults()
    {
        var a = new Matrix<double>(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var b = new Matrix<double>(new double[,] { { 7, 8 }, { 9, 10 }, { 11, 12 } });
        Assert.Equal(new double[,] { { 58, 64 }, { 139, 154 } }, (a * b).ToArray());
        Assert.Equal(XMath.Multiply(XMath.Transpose(a), a).ToArray(), XMath.Multiply(a.T, a).ToArray());
        Assert.Equal(XMath.Multiply(a, XMath.Transpose(a)).ToArray(), XMath.Multiply(a, a.T).ToArray());
        Assert.Equal(Blas.Norm2(a.AsSpan()), a.FrobeniusNorm());
        var x = new double[] { 1, 2, 3 }.ToVector();
        double[] expected = new double[2];
        Blas.Gemv(BlasTranspose.None, 2, 3, 1, a.AsSpan(), 2, x.AsSpan(), 0, expected);
        Assert.Equal(expected, (a * x).ToArray());
        Assert.Equal(Blas.Dot(x.AsSpan(), x.AsSpan()), x.Dot(x));
        Assert.Equal(Blas.Norm2(x.AsSpan()), x.Norm2());
        Assert.Equal(new double[,] { { 2, 4, 6 }, { 8, 10, 12 } }, (a + a).ToArray());
        Assert.Equal(new double[2, 3], (a - a).ToArray());
        Assert.Equal((2 * a).ToArray(), (a * 2).ToArray());
        Assert.Equal(new double[] { 2, 4, 6 }, (x + x).ToArray());
        Assert.Equal(new double[3], (x - x).ToArray());
        Assert.Equal((2 * x).ToArray(), (x * 2).ToArray());
    }

    [Fact]
    public void OtherScalarTypesAndEmptyProductsUseExistingKernels()
    {
        var a = new Matrix<float>(new float[,] { { 1, 2 }, { 3, 4 } });
        Assert.Equal(new float[,] { { 7, 10 }, { 15, 22 } }, (a * a).ToArray());
        Assert.Equal(new int[,] { { 7, 10 }, { 15, 22 } }, (a.ConvertChecked<int>() * a.ConvertChecked<int>()).ToArray());
        var x = new float[] { 3, 4 }.ToVector();
        Assert.Equal(25f, x.Dot(x));
        Assert.Equal(5f, x.Norm2());
        var c = new Complex[,] { { new(1, 1), new(2, -1) }, { new(3, 2), new(-1, 1) } }.ToMatrix();
        var y = new Complex[] { new(2, 1), new(-1, 3) }.ToVector();
        var expected = new Complex[2];
        Blas.Gemv(BlasTranspose.None, 2, 2, Complex.One, c.AsSpan(), 2, y.AsSpan(), Complex.Zero, expected);
        Assert.Equal(expected, (c * y).ToArray());
        Assert.Equal(Blas.Dot(y.AsSpan(), y.AsSpan()), y.Dot(y));
        Assert.Equal(Blas.DotConjugate(y.AsSpan(), y.AsSpan()), y.DotConjugate(y));
        Assert.Equal(Blas.Norm2(y.AsSpan()), y.Norm2());
        Assert.Equal(c.ToArray(), (c * Matrix<Complex>.Identity(2)).ToArray());
        Assert.Equal(new double[2, 3], (new Matrix<double>(2, 0) * new Matrix<double>(0, 3)).ToArray());
        Assert.Equal(new float[2, 3], (new Matrix<float>(2, 0) * new Matrix<float>(0, 3)).ToArray());
        Assert.Equal(new double[] { 0, 0 }, (new Matrix<double>(2, 0) * new DoubleVector(0)).ToArray());
        Assert.Throws<ArgumentException>(() => a * new Matrix<float>(3, 2));
        Assert.Throws<ArgumentException>(() => a * new Tuna.LinearAlgebra.Vector<float>(3));
        Assert.Throws<ArgumentException>(() => a + new Matrix<float>(1, 4));
        Assert.Throws<ArgumentException>(() => x + new Tuna.LinearAlgebra.Vector<float>(1));
    }

    [Fact]
    public void DecompositionsKeepTheInputUnlessAskedToOverwriteIt()
    {
        var a = new Matrix<double>(new double[,] { { 4, 1, 0.5 }, { 1, 3, 0.25 }, { 0.5, 0.25, 2 } });
        var original = a.ToArray();
        var cholesky = a.Cholesky();
        var qr = a.Qr();
        var svd = a.Svd();
        var symmetric = a.SymmetricEigen();
        var eigen = a.Eigen();
        Assert.Equal(original, a.ToArray());

        // 入力を作業領域に使っても、結果はビット単位で同じ
        Assert.Equal(cholesky.L.ToArray(), CholeskyDecomposition.Compute(a.Clone(), a.Clone()).L.ToArray());
        Assert.Equal(qr.R.ToArray(), QrDecomposition.Compute(a.Clone(), overwriteInput: true).R.ToArray());
        Assert.Equal(svd.Values, SingularValueDecomposition.Compute(a.Clone(), overwriteInput: true).Values);
        Assert.Equal(symmetric.Values, SymmetricEigenDecomposition.Compute(a.Clone(), overwriteInput: true).Values);
        Assert.Equal(eigen.Values, EigenDecomposition.Compute(a.Clone(), overwriteInput: true).Values);
        var b = a.Clone();
        QrDecomposition.Compute(b, overwriteInput: true);
        Assert.NotEqual(original, b.ToArray());
    }

    [Theory]
    [InlineData(17, 9, 33)]
    [InlineData(33, 31, 257)]
    public void TransposedViewsMatchCopiedTransposesWithinRounding(int m, int n, int k)
    {
        var random = new Random(9123);
        var a = new Matrix<double>(m, k);
        var b = new Matrix<double>(k, n);
        for (int i = 0; i < a.AsSpan().Length; i++)
        {
            a.AsSpan()[i] = random.NextDouble() - 0.5;
        }

        for (int i = 0; i < b.AsSpan().Length; i++)
        {
            b.AsSpan()[i] = random.NextDouble() - 0.5;
        }
        // 転置したコピーの転置ビューは GEMM の転置フラグ経路を通る。小さい行列の経路では加算の順序が変わるので丸め誤差の範囲で比べる
        var expected = (a * b).AsSpan().ToArray();
        foreach (var actual in new[]
        {
            XMath.Multiply(XMath.Transpose(a).T, b).AsSpan().ToArray(),
            XMath.Multiply(a, XMath.Transpose(b).T).AsSpan().ToArray(),
        })
        {
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], actual[i], 4 * k * 2.220446049250313e-16);
            }
        }
    }
}
