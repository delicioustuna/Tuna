using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "WrightOmega")]
public class WrightOmegaTests
{

    // 指数関数の上下限・級数/ハレー法の切替・巨大入力。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-745.0, 5e-324, 1e-323)]
    [InlineData(-740.0, 4.2e-322, 1e-323)]
    [InlineData(-40.00000000000001, 4.248354255291559e-18, 0.0)]
    [InlineData(-40.0, 4.248354255291589e-18, 0.0)]
    [InlineData(-39.99999999999999, 4.248354255291619e-18, 0.0)]
    [InlineData(-2.0, 0.12002823898764123, 0.0)]
    [InlineData(0.0, 0.5671432904097838, 0.0)]
    [InlineData(0.9999999999999999, 1.0, 0.0)]
    [InlineData(1.0, 1.0, 0.0)]
    [InlineData(1.0000000000000002, 1.0000000000000002, 0.0)]
    [InlineData(10.0, 7.929420095019697, 0.0)]
    [InlineData(710.0, 703.4440117119545, 0.0)]
    [InlineData(10000.0, 9990.790580994251, 0.0)]
    [InlineData(1e+100, 1e+100, 0.0)]
    public void WrightOmega_MatchesIndependentReferences(double x, double expected, double absoluteTolerance)
    {
        double actual = XMath.Special.WrightOmega(x);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12, absoluteTolerance: absoluteTolerance);
            if (Math.Abs(expected) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expected), Math.Sign(actual));
            }
        }
    }

    // 各引数のNaNを独立に検査。
    [Theory]
    [InlineData(double.NaN)]
    public void WrightOmega_RejectsNaN(double x)
    {
        double actual = XMath.Special.WrightOmega(x);
        Assert.True(double.IsNaN(actual));
    }

    [Fact]
    public void LimitsAndDefiningEquation()
    {
        Assert.Equal(0, XMath.Special.WrightOmega(double.NegativeInfinity));
        Assert.Equal(double.PositiveInfinity, XMath.Special.WrightOmega(double.PositiveInfinity));
        Assert.Equal(0, XMath.Special.WrightOmega(-1000));
        Assert.Equal(1, XMath.Special.WrightOmega(1));
        foreach (double x in new[] { -10.0, -1.0, 2.0, 20.0, 1000.0 })
        {
            double w = XMath.Special.WrightOmega(x);
            SpecialFunctionAssert.Close(x, w + Math.Log(w), relativeTolerance: 2e-12);
        }
    }
}
