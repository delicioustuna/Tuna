using Tuna.LinearAlgebra;
using static Tuna.Tests.LinearAlgebra.LinearAlgebraTestUtilities;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>行列型と行列積（パッキング・マイクロカーネルの端の処理を含む）の動作確認</summary>
[Trait("LinearAlgebra", "Matrix")]
public class MatrixTests
{
    /// <summary>二次元配列・列優先・行優先との相互変換が一致する</summary>
    [Fact]
    public void ConversionsPreserveLayout()
    {
        var values = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };
        var a = new Matrix<double>(values);
        Assert.Equal(2, a.Rows);
        Assert.Equal(3, a.Columns);
        Assert.Equal(6.0, a[1, 2]);
        Assert.Equal(new double[] { 1, 4, 2, 5, 3, 6 }, a.AsSpan().ToArray());
        Assert.Equal(values, a.ToArray());
        Assert.Equal(values, Matrix<double>.FromRowMajor(2, 3, [1, 2, 3, 4, 5, 6]).ToArray());
        Assert.Equal(values, Matrix<double>.FromColumnMajor(2, 3, [1, 4, 2, 5, 3, 6]).ToArray());
        Assert.Equal(new double[,] { { 1, 4 }, { 2, 5 }, { 3, 6 } }, XMath.Transpose(a).ToArray());
        Assert.Throws<ArgumentOutOfRangeException>(() => a[2, 0]);
        Assert.Throws<ArgumentException>(() => Matrix<double>.FromColumnMajor(2, 2, [1, 2, 3]));
    }

    /// <summary>フロベニウスノルムはオーバーフロー・アンダーフローしない</summary>
    [Fact]
    public void FrobeniusNormAvoidsOverflow()
    {
        Assert.Equal(5e300, Matrix<double>.FromColumnMajor(2, 1, [3e300, 4e300]).FrobeniusNorm(), 1e286);
        Assert.Equal(5e-300, Matrix<double>.FromColumnMajor(2, 1, [3e-300, 4e-300]).FrobeniusNorm(), 1e-314);
        Assert.Equal(0.0, new Matrix<double>(3, 3).FrobeniusNorm());
    }

    /// <summary>行列積が単純な三重ループと丸め誤差の範囲で一致する（転置の4通り、端数のある大きさ）</summary>
    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(3, 5, 7)]
    [InlineData(16, 8, 16)]
    [InlineData(17, 9, 33)]
    [InlineData(33, 31, 257)]
    [InlineData(150, 70, 300)]
    [InlineData(300, 3, 5)]
    [InlineData(5, 300, 3)]
    public void MultiplyMatchesNaive(int m, int n, int k)
    {
        var a = Random(m, k, 1);
        var b = Random(k, n, 2);
        var expected = MultiplyNaive(a, b);
        double scale = Norm(a) * Norm(b);
        AssertRatio(Difference(expected, a * b), scale, k, "AB");
        AssertRatio(Difference(expected, XMath.Multiply(XMath.Transpose(a).T, b)), scale, k, "(Aᵀ)ᵀB");
        AssertRatio(Difference(expected, XMath.Multiply(a, XMath.Transpose(b).T)), scale, k, "A(Bᵀ)ᵀ");
        AssertRatio(Difference(XMath.Transpose(expected), XMath.Multiply(b.T, a.T)), scale, k, "BᵀAᵀ");
    }

    /// <summary>
    /// 小さな積（<c>m·k ≤ 576</c>、A が列優先）は、各要素を k の順に FMA で足す（<c>c = fma(a[i, p], b[p, j], c)</c>）。
    /// SIMD のレーン（8・4・2 行）と端の行で同じ規則になり、B の転置ビューでも同じビットになる。FMA の命令がない環境では GEMM の経路のままなので確かめない
    /// </summary>
    [Fact]
    public void SmallProductsAccumulateEachElementInOrderWithFusedMultiplyAdd()
    {
        if (!System.Runtime.Intrinsics.X86.Fma.IsSupported && !System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.IsSupported)
        {
            return;
        }

        foreach (int m in new[] { 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 24 })
        {
            foreach (int k in new[] { 1, 2, 5, 9, 24 })
            {
                const int n = 3;
                var a = Random(m, k, 100 + m * 31 + k);
                var b = Random(k, n, 200 + m * 31 + k);
                var expected = new double[m, n];
                for (int j = 0; j < n; j++)
                {
                    for (int i = 0; i < m; i++)
                    {
                        double sum = 0;
                        for (int p = 0; p < k; p++)
                        {
                            sum = Math.FusedMultiplyAdd(a[i, p], b[p, j], sum);
                        }

                        expected[i, j] = sum;
                    }
                }

                Assert.Equal(expected, (a * b).ToArray());
                Assert.Equal(expected, XMath.Multiply(a, XMath.Transpose(b).T).ToArray());
            }
        }
    }

    /// <summary>行列とベクトルの積、和、差、スカラー倍</summary>
    [Fact]
    public void ElementwiseOperators()
    {
        var a = new Matrix<double>(new double[,] { { 1, 2 }, { 3, 4 } });
        var b = new Matrix<double>(new double[,] { { 5, 6 }, { 7, 8 } });
        Assert.Equal(new double[,] { { 6, 8 }, { 10, 12 } }, (a + b).ToArray());
        Assert.Equal(new double[,] { { -4, -4 }, { -4, -4 } }, (a - b).ToArray());
        Assert.Equal(new double[,] { { 2, 4 }, { 6, 8 } }, (2.0 * a).ToArray());
        Assert.Equal(new double[] { 5, 11 }, (a * new double[] { 1, 2 }.ToVector()).ToArray());
        Assert.Equal(new double[,] { { 19, 22 }, { 43, 50 } }, (a * b).ToArray());
        Assert.Throws<ArgumentException>(() => a * new Matrix<double>(3, 3));
    }
}
