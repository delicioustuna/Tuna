using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "Scorer")]
public class ScorerTests
{

    // 正負の漸近域・原点・増大/減衰解。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-50.0, -0.14351624800962248)]
    [InlineData(-20.0, -0.2160508346367536)]
    [InlineData(-12.0, -0.32221537922982774)]
    [InlineData(-1.0, -0.11667221729601528)]
    [InlineData(0.0, 0.20497554248200026)]
    [InlineData(1.0, 0.23521843981043794)]
    [InlineData(10.0, 0.03189600510067959)]
    [InlineData(20.0, 0.015919483200561038)]
    [InlineData(50.0, 0.006366299599144166)]
    [InlineData(100.0, 0.0031831052281629615)]
    public void ScorerGi_MatchesIndependentReferences(double x, double expected)
    {
        double actual = XMath.Special.ScorerGi(x);
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
    public void ScorerGi_RejectsNaN(double x)
    {
        double actual = XMath.Special.ScorerGi(x);
        Assert.True(double.IsNaN(actual));
    }

    // 正負の漸近域・原点・増大/減衰解。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-50.0, 0.006366095880802406)]
    [InlineData(-20.0, 0.015911525314102233)]
    [InlineData(-12.0, 0.026495467151754687)]
    [InlineData(-1.0, 0.22066960679295988)]
    [InlineData(0.0, 0.4099510849640005)]
    [InlineData(1.0, 0.9722051551424333)]
    [InlineData(10.0, 455641153.5163291)]
    [InlineData(20.0, 2.103765049651104e+25)]
    [InlineData(50.0, 4.9090996994442195e+101)]
    [InlineData(100.0, 6.041223996670201e+288)]
    public void ScorerHi_MatchesIndependentReferences(double x, double expected)
    {
        double actual = XMath.Special.ScorerHi(x);
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
    public void ScorerHi_RejectsNaN(double x)
    {
        double actual = XMath.Special.ScorerHi(x);
        Assert.True(double.IsNaN(actual));
    }

    [Theory]
    [InlineData(-10.0)]
    [InlineData(0.0)]
    [InlineData(10.0)]
    public void SumIsAiryBiAndLimits(double x)
    {
        SpecialFunctionAssert.Close(XMath.Special.AiryBi(x), XMath.Special.ScorerGi(x) + XMath.Special.ScorerHi(x), relativeTolerance: 2e-12);
        Assert.Equal(0, XMath.Special.ScorerGi(double.PositiveInfinity));
        Assert.Equal(0, XMath.Special.ScorerHi(double.NegativeInfinity));
        Assert.Equal(double.PositiveInfinity, XMath.Special.ScorerHi(double.PositiveInfinity));
    }
}
