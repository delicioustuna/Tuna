using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "InverseDigamma")]
public class InverseDigammaTests
{

    // 初期値切替の隣接値・小さい正の解・expの上限付近。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-1e+200, 1e-200)]
    [InlineData(-100.0, 0.010056395666750782)]
    [InlineData(-10.0, 0.10435719877011651)]
    [InlineData(-2.2200000000000006, 0.4522984965107054)]
    [InlineData(-2.22, 0.45229849651070547)]
    [InlineData(-2.2199999999999998, 0.4522984965107055)]
    [InlineData(-0.5772156649015329, 1.0)]
    [InlineData(0.0, 1.4616321449683622)]
    [InlineData(1.0, 3.2031714683769312)]
    [InlineData(10.0, 22026.965792915053)]
    [InlineData(100.0, 2.6881171418161356e+43)]
    [InlineData(700.0, 1.0142320547350045e+304)]
    public void InverseDigamma_MatchesIndependentReferences(double y, double expected)
    {
        double actual = XMath.Special.InverseDigamma(y);
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
    public void InverseDigamma_RejectsNaN(double y)
    {
        double actual = XMath.Special.InverseDigamma(y);
        Assert.True(double.IsNaN(actual));
    }

    [Fact]
    public void LimitsAndMonotonicity()
    {
        Assert.Equal(0, XMath.Special.InverseDigamma(double.NegativeInfinity));
        Assert.Equal(double.PositiveInfinity, XMath.Special.InverseDigamma(double.PositiveInfinity));
        Assert.Equal(double.PositiveInfinity, XMath.Special.InverseDigamma(710));
        double previous = 0;
        foreach (double y in new[] { -100.0, -3.0, -2.22, -1.0, 0.0, 10.0, 100.0 })
        {
            double x = XMath.Special.InverseDigamma(y);
            Assert.True(double.IsFinite(x) && x > previous);
            previous = x;
        }
    }
}
