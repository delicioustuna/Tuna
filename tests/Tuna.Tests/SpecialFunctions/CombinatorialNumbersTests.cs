using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "CombinatorialNumbers")]
public class CombinatorialNumbersTests
{

    // 厳密整数からbinary64への丸め・表の上限近く。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0, 1.0)]
    [InlineData(1, 0.0)]
    [InlineData(2, -1.0)]
    [InlineData(4, 5.0)]
    [InlineData(10, -50521.0)]
    [InlineData(20, 370371188237525.0)]
    [InlineData(50, -6.053285248188622e+54)]
    [InlineData(100, 2.9035283466610976e+138)]
    [InlineData(180, 1.2773316636719806e+294)]
    public void EulerNumber_MatchesIndependentReferences(int n, double expected)
    {
        double actual = XMath.Special.EulerNumber(n);
        Assert.Equal(expected, actual);
    }

    // 厳密整数からbinary64への丸め・表の上限近く。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0, 1.0)]
    [InlineData(1, 1.0)]
    [InlineData(2, 2.0)]
    [InlineData(4, 15.0)]
    [InlineData(10, 115975.0)]
    [InlineData(20, 51724158235372.0)]
    [InlineData(50, 1.8572426877107827e+47)]
    [InlineData(100, 4.758539127676483e+115)]
    [InlineData(218, 6.101309833875322e+306)]
    public void BellNumber_MatchesIndependentReferences(int n, double expected)
    {
        double actual = XMath.Special.BellNumber(n);
        Assert.Equal(expected, actual);
    }

    // 符号・端点・対角付近・直接和/漸化式の領域。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0, 0, 1.0)]
    [InlineData(5, 0, 0.0)]
    [InlineData(5, 1, 24.0)]
    [InlineData(5, 3, 35.0)]
    [InlineData(5, 5, 1.0)]
    [InlineData(5, 6, 0.0)]
    [InlineData(20, 10, 381922055502195.0)]
    [InlineData(50, 45, -17392967051250.0)]
    [InlineData(80, 20, 4.1386319716598104e+109)]
    [InlineData(200, 190, 1.976590735776009e+36)]
    [InlineData(1000, 2, double.PositiveInfinity)]
    public void StirlingFirstKind_MatchesIndependentReferences(int n, int k, double expected)
    {
        double actual = XMath.Special.StirlingFirstKind(n, k);
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

    // 符号・端点・対角付近・直接和/漸化式の領域。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0, 0, 1.0)]
    [InlineData(5, 0, 0.0)]
    [InlineData(5, 1, 1.0)]
    [InlineData(5, 3, 25.0)]
    [InlineData(5, 5, 1.0)]
    [InlineData(5, 6, 0.0)]
    [InlineData(20, 10, 5917584964655.0)]
    [InlineData(50, 45, 13132829288250.0)]
    [InlineData(80, 20, 3.521648056479197e+85)]
    [InlineData(200, 190, 1.4529713736812635e+36)]
    [InlineData(1000, 2, 5.357543035931337e+300)]
    public void StirlingSecondKind_MatchesIndependentReferences(int n, int k, double expected)
    {
        double actual = XMath.Special.StirlingSecondKind(n, k);
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

    [Fact]
    public void InvalidArgumentsAndOverflowSigns()
    {
        Assert.True(double.IsNaN(XMath.Special.EulerNumber(-1)));
        Assert.True(double.IsNaN(XMath.Special.BellNumber(-1)));
        Assert.True(double.IsNaN(XMath.Special.StirlingFirstKind(-1, 1)));
        Assert.True(double.IsNaN(XMath.Special.StirlingSecondKind(2, -1)));
        Assert.Equal(0, XMath.Special.EulerNumber(189));
        Assert.Equal(double.PositiveInfinity, XMath.Special.EulerNumber(188));
        Assert.Equal(double.NegativeInfinity, XMath.Special.EulerNumber(190));
        Assert.Equal(double.PositiveInfinity, XMath.Special.BellNumber(219));
        Assert.Equal(0, XMath.Special.StirlingFirstKind(3, 4));
        Assert.Equal(0, XMath.Special.StirlingSecondKind(3, 4));
    }

    [Theory]
    [InlineData(5, 2)]
    [InlineData(10, 8)]
    [InlineData(20, 10)]
    public void StirlingRecurrences(int n, int k)
    {
        SpecialFunctionAssert.Close(XMath.Special.StirlingSecondKind(n - 1, k - 1) + k * XMath.Special.StirlingSecondKind(n - 1, k), XMath.Special.StirlingSecondKind(n, k), relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(XMath.Special.StirlingFirstKind(n - 1, k - 1) - (n - 1) * XMath.Special.StirlingFirstKind(n - 1, k), XMath.Special.StirlingFirstKind(n, k), relativeTolerance: 2e-12);
    }
}
