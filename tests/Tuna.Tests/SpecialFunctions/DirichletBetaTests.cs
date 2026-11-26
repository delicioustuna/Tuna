using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "DirichletBeta")]
public class DirichletBetaTests
{

    // 反射・負整数の零点・s=1の可除特異点。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-180.0, 6.386658318359903e+293)]
    [InlineData(-10.0, -25260.5)]
    [InlineData(-3.0, 0.0)]
    [InlineData(-2.0, -0.5)]
    [InlineData(0.0, 0.5)]
    [InlineData(0.49999999999999994, 0.6676914571896092)]
    [InlineData(0.5, 0.6676914571896092)]
    [InlineData(0.5000000000000001, 0.6676914571896092)]
    [InlineData(0.9999999999999999, 0.7853981633974483)]
    [InlineData(1.0, 0.7853981633974483)]
    [InlineData(1.0000000000000002, 0.7853981633974484)]
    [InlineData(2.0, 0.915965594177219)]
    [InlineData(3.0, 0.9689461462593694)]
    [InlineData(20.0, 0.9999999997132133)]
    [InlineData(60.0, 1.0)]
    [InlineData(61.0, 1.0)]
    public void DirichletBeta_MatchesIndependentReferences(double s, double expected)
    {
        double actual = XMath.Special.DirichletBeta(s);
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
    public void DirichletBeta_RejectsNaN(double s)
    {
        double actual = XMath.Special.DirichletBeta(s);
        Assert.True(double.IsNaN(actual));
    }

    [Fact]
    public void ClosedFormsTrivialZerosAndLimits()
    {
        SpecialFunctionAssert.Close(.5, XMath.Special.DirichletBeta(0), relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(Math.PI / 4, XMath.Special.DirichletBeta(1), relativeTolerance: 2e-12);
        Assert.Equal(0, XMath.Special.DirichletBeta(-1));
        Assert.Equal(0, XMath.Special.DirichletBeta(-101));
        Assert.Equal(1, XMath.Special.DirichletBeta(double.PositiveInfinity));
        Assert.True(double.IsNaN(XMath.Special.DirichletBeta(double.NegativeInfinity)));
    }

    // u = 1 - s が丸められる帯（-128 < s ≤ -127）で Γ(u)(2/π)ᵘ の ψ 補正が効いていることを確認する（3e-15 は補正検出用）。
    // 参照値は mpmath 1.4.1 の dirichlet を binary64 入力のまま 300/500 bit で評価して一致を確認。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-127.94437865599103, 1.5088631299178382e+190)]
    public void DirichletBeta_RoundedReflectionArgument_IsCorrected(double s, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.DirichletBeta(s), relativeTolerance: 3e-15);
    }
}
