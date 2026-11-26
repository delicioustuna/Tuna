using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "Mathieu")]
public class MathieuTests
{

    // 高精度の密行列固有値/固有ベクトル。24/32項の打切りも照合。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0, 1.0, -0.4551386041074135)]
    [InlineData(1, 1.0, 1.8591080725143634)]
    [InlineData(2, -2.0, 5.1726651333582945)]
    [InlineData(5, 10.0, 27.703768733939278)]
    public void MathieuA_MatchesIndependentReferences(int n, double q, double expected)
    {
        double actual = XMath.Special.MathieuA(n, q);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12);
            if (Math.Abs(expected) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expected), Math.Sign(actual));
            }
        }
    }

    // 各引数のNaNを独立に検査。
    [Theory]
    [InlineData(0, double.NaN)]
    public void MathieuA_RejectsNaN(int n, double q)
    {
        double actual = XMath.Special.MathieuA(n, q);
        Assert.True(double.IsNaN(actual));
    }

    // 高精度の密行列固有値/固有ベクトル。24/32項の打切りも照合。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1, 1.0, -0.11024881699209517)]
    [InlineData(2, -2.0, 3.6722327064971907)]
    [InlineData(5, 10.0, 26.76642636048006)]
    public void MathieuB_MatchesIndependentReferences(int n, double q, double expected)
    {
        double actual = XMath.Special.MathieuB(n, q);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12);
            if (Math.Abs(expected) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expected), Math.Sign(actual));
            }
        }
    }

    // 各引数のNaNを独立に検査。
    [Theory]
    [InlineData(1, double.NaN)]
    public void MathieuB_RejectsNaN(int n, double q)
    {
        double actual = XMath.Special.MathieuB(n, q);
        Assert.True(double.IsNaN(actual));
    }

    // 高精度の密行列固有値/固有ベクトル。24/32項の打切りも照合。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0, 1.0, 0.3, 0.4270532603468534)]
    [InlineData(1, 1.0, 0.3, 0.8597444763384642)]
    [InlineData(2, -2.0, 0.3, 0.4357975025415299)]
    [InlineData(5, 10.0, 0.3, 0.8183430397503374)]
    public void MathieuCe_MatchesIndependentReferences(int n, double q, double x, double expected)
    {
        double actual = XMath.Special.MathieuCe(n, q, x);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12);
            if (Math.Abs(expected) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expected), Math.Sign(actual));
            }
        }
    }

    // 各引数のNaNを独立に検査。
    [Theory]
    [InlineData(0, double.NaN, 0.3)]
    [InlineData(0, 1.0, double.NaN)]
    public void MathieuCe_RejectsNaN(int n, double q, double x)
    {
        double actual = XMath.Special.MathieuCe(n, q, x);
        Assert.True(double.IsNaN(actual));
    }

    // 高精度の密行列固有値/固有ベクトル。24/32項の打切りも照合。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1, 1.0, 0.3, 0.21217807304059402)]
    [InlineData(2, -2.0, 0.3, 0.7178977535337573)]
    [InlineData(5, 10.0, 0.3, 0.906282068799812)]
    public void MathieuSe_MatchesIndependentReferences(int n, double q, double x, double expected)
    {
        double actual = XMath.Special.MathieuSe(n, q, x);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12);
            if (Math.Abs(expected) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expected), Math.Sign(actual));
            }
        }
    }

    // 各引数のNaNを独立に検査。
    [Theory]
    [InlineData(1, double.NaN, 0.3)]
    [InlineData(1, 1.0, double.NaN)]
    public void MathieuSe_RejectsNaN(int n, double q, double x)
    {
        double actual = XMath.Special.MathieuSe(n, q, x);
        Assert.True(double.IsNaN(actual));
    }

    [Theory]
    [InlineData(1, .2)]
    [InlineData(2, 1.0)]
    [InlineData(5, -1.0)]
    public void ZeroParameterIsTrigonometric(int n, double x)
    {
        Assert.Equal((double)n * n, XMath.Special.MathieuA(n, 0));
        Assert.Equal((double)n * n, XMath.Special.MathieuB(n, 0));
        SpecialFunctionAssert.Close(Math.Cos(n * x), XMath.Special.MathieuCe(n, 0, x), relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(Math.Sin(n * x), XMath.Special.MathieuSe(n, 0, x), relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(1 / Math.Sqrt(2), XMath.Special.MathieuCe(0, 0, x), relativeTolerance: 2e-12);
    }

    [Fact]
    public void ParameterReflectionAndDomains()
    {
        SpecialFunctionAssert.Close(XMath.Special.MathieuA(2, 3), XMath.Special.MathieuA(2, -3), relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(XMath.Special.MathieuA(1, 3), XMath.Special.MathieuB(1, -3), relativeTolerance: 2e-12);
        Assert.True(double.IsNaN(XMath.Special.MathieuA(-1, 1)));
        Assert.True(double.IsNaN(XMath.Special.MathieuB(0, 1)));
        Assert.True(double.IsNaN(XMath.Special.MathieuA(100001, 1)));
        Assert.True(double.IsNaN(XMath.Special.MathieuA(1, 1e13)));
        Assert.True(double.IsNaN(XMath.Special.MathieuCe(1, 1, double.PositiveInfinity)));
        Assert.True(double.IsNaN(XMath.Special.MathieuSe(0, 1, .3)));
    }
}
