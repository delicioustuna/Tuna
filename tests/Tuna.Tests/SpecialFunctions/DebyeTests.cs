using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "Debye")]
public class DebyeTests
{

    // 定義積分・小引数・負引数・大引数。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1, 0.01, 0.997502777775)]
    [InlineData(2, 0.5, 0.8437211933472536)]
    [InlineData(3, 1.0, 0.6744155640778147)]
    [InlineData(5, 5.0, 0.07686342544180193)]
    [InlineData(10, 10.0, 0.0015148395086342053)]
    [InlineData(3, 50.0, 0.0001558545456544039)]
    [InlineData(1, -2.0, 1.60694728460981)]
    [InlineData(3, -5.0, 3.867597411799934)]
    [InlineData(20, 0.2, 0.9077903575668472)]
    public void Debye_MatchesIndependentReferences(int n, double x, double expected)
    {
        double actual = XMath.Special.Debye(n, x);
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
    public void Debye_RejectsNaN(int n, double x)
    {
        double actual = XMath.Special.Debye(n, x);
        Assert.True(double.IsNaN(actual));
    }

    [Theory]
    [InlineData(1, .2)]
    [InlineData(3, 2.0)]
    [InlineData(10, 10.0)]
    public void NegativeArgumentRelationAndLimits(int n, double x)
    {
        SpecialFunctionAssert.Close(XMath.Special.Debye(n, x) + n * x / (n + 1), XMath.Special.Debye(n, -x), relativeTolerance: 2e-12);
        Assert.Equal(1, XMath.Special.Debye(n, 0));
        Assert.Equal(0, XMath.Special.Debye(n, double.PositiveInfinity));
        Assert.Equal(double.PositiveInfinity, XMath.Special.Debye(n, double.NegativeInfinity));
        Assert.True(double.IsNaN(XMath.Special.Debye(0, x)));
    }
}
