using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "Hypergeometric2F1")]
public class Hypergeometric2F1Tests
{

    // 負引数変換・x=1のGauss公式・終端多項式・接続公式の対数分岐。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1.0, 1.0, 2.0, 0.5, 1.3862943611198906)]
    [InlineData(0.3, 0.7, 1.4, -10.0, 0.6353150973940093)]
    [InlineData(2.0, 3.0, 5.0, 0.9, 8.561538392187561)]
    [InlineData(2.0, 3.0, 6.0, 1.0, 10.0)]
    [InlineData(-3.0, 0.5, 2.0, 2.0, 0.375)]
    [InlineData(-2.0, 1.0, -3.0, 0.2, 1.1466666666666667)]
    [InlineData(0.5, 0.5, 1.0, 0.99, 2.3527158167797424)]
    [InlineData(0.3, 0.7, 1.4, 0.9999999999999999, 1.593718359223427)]
    [InlineData(0.5, 1.5, 2.0, -0.5, 0.8557037256498209)]
    public void Hypergeometric2F1_MatchesIndependentReferences(double a, double b, double c, double x, double expected)
    {
        double actual = XMath.Special.Hypergeometric2F1(a, b, c, x);
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
    [InlineData(double.NaN, 1.0, 2.0, 0.5)]
    [InlineData(1.0, double.NaN, 2.0, 0.5)]
    [InlineData(1.0, 1.0, double.NaN, 0.5)]
    [InlineData(1.0, 1.0, 2.0, double.NaN)]
    public void Hypergeometric2F1_RejectsNaN(double a, double b, double c, double x)
    {
        double actual = XMath.Special.Hypergeometric2F1(a, b, c, x);
        Assert.True(double.IsNaN(actual));
    }

    [Theory]
    [InlineData(.2)]
    [InlineData(-5.0)]
    [InlineData(.9)]
    public void ClosedFormsAndParameterSymmetry(double x)
    {
        SpecialFunctionAssert.Close(Math.Pow(1 - x, -.3), XMath.Special.Hypergeometric2F1(.3, 2, 2, x), relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(XMath.Special.Hypergeometric2F1(.3, .7, 1.4, x), XMath.Special.Hypergeometric2F1(.7, .3, 1.4, x), relativeTolerance: 2e-12);
        Assert.True(double.IsNaN(XMath.Special.Hypergeometric2F1(1, 1, 0, x)));
        Assert.True(double.IsNaN(XMath.Special.Hypergeometric2F1(.3, .7, 2, 1.1)));
    }
}
