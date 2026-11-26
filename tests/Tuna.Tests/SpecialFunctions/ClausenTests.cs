using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "Clausen")]
public class ClausenTests
{

    // 奇関数・零点の近く・周期還元。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-10.0, 0.3907164760868021)]
    [InlineData(-1.0, -1.0139591323607684)]
    [InlineData(-1e-20, -4.705170185988091e-19)]
    [InlineData(0.0, 0.0)]
    [InlineData(1e-20, 4.705170185988091e-19)]
    [InlineData(0.5, 0.8483118777036792)]
    [InlineData(1.5707963267948966, 0.915965594177219)]
    [InlineData(3.0, 0.09802620939130142)]
    [InlineData(3.141592653589793, 8.488604760107495e-17)]
    [InlineData(6.0, -0.6407826657017233)]
    [InlineData(20.0, 1.0105014481412877)]
    public void Clausen_MatchesIndependentReferences(double theta, double expected)
    {
        double actual = XMath.Special.Clausen(theta);
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
    [InlineData(double.NaN)]
    public void Clausen_RejectsNaN(double theta)
    {
        double actual = XMath.Special.Clausen(theta);
        Assert.True(double.IsNaN(actual));
    }

    [Theory]
    [InlineData(.1)]
    [InlineData(1.0)]
    [InlineData(2.0)]
    public void OddAndPeriodic(double x)
    {
        Assert.Equal(-XMath.Special.Clausen(x), XMath.Special.Clausen(-x));
        SpecialFunctionAssert.Close(XMath.Special.Clausen(x), XMath.Special.Clausen(x + 2 * Math.PI), relativeTolerance: 2e-12);
        Assert.Equal(0, XMath.Special.Clausen(0));
        Assert.True(double.IsNaN(XMath.Special.Clausen(double.PositiveInfinity)));
    }
    // 巨大な周期数でも、Cl₂ の値域と位相を保持する。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditIdentities20261102")]
    [Theory]
    [InlineData(-1e+16, -0.5893821278800236)]
    [InlineData(1e+16, 0.5893821278800236)]
    [InlineData(-1e+18, 0.9534236178524811)]
    [InlineData(1e+18, -0.9534236178524811)]
    [InlineData(-1e+20, 0.9549568361169309)]
    [InlineData(1e+20, -0.9549568361169309)]
    public void Clausen_LargeFiniteAnglesRemainBounded(double theta, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Clausen(theta), relativeTolerance: 2e-12);
    }
}
