using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "Lerch")]
public class LerchTests
{

    // z=0/1の閉形式・負の枝・高nome・負次数。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.0, 2.0, 0.5, 4.0)]
    [InlineData(0.3, 2.0, 1.0, 1.0870983669182535)]
    [InlineData(-0.5, 0.7, 0.3, 2.0078253844655185)]
    [InlineData(0.99, 3.0, 2.0, 0.19980913544029888)]
    [InlineData(1.0, 2.0, 0.5, 4.934802200544679)]
    [InlineData(-2.0, 0.5, 1.0, 0.44564435577606165)]
    [InlineData(0.2, -3.0, 2.0, 17.4609375)]
    [InlineData(-0.9, 4.0, 0.1, 9999.420752457469)]
    public void LerchPhi_MatchesIndependentReferences(double z, double s, double a, double expected)
    {
        double actual = XMath.Special.LerchPhi(z, s, a);
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
    [InlineData(double.NaN, 2.0, 0.5)]
    [InlineData(0.0, double.NaN, 0.5)]
    [InlineData(0.0, 2.0, double.NaN)]
    public void LerchPhi_RejectsNaN(double z, double s, double a)
    {
        double actual = XMath.Special.LerchPhi(z, s, a);
        Assert.True(double.IsNaN(actual));
    }

    [Theory]
    [InlineData(.2, .5)]
    [InlineData(-.5, 2.0)]
    public void PolylogarithmRelationAndDomains(double z, double s)
    {
        SpecialFunctionAssert.Close(XMath.Special.Polylogarithm(s, z) / z, XMath.Special.LerchPhi(z, s, 1), relativeTolerance: 2e-12);
        Assert.True(double.IsNaN(XMath.Special.LerchPhi(.3, 2, 0)));
        Assert.True(double.IsNaN(XMath.Special.LerchPhi(1.1, 2, 1)));
        SpecialFunctionAssert.Close(4, XMath.Special.LerchPhi(0, 2, .5), relativeTolerance: 2e-12);
    }
}
