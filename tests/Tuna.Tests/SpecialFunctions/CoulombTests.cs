using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "Coulomb")]
public class CoulombTests
{

    // 自由波・非整数角運動量・障壁の内外・遠方の振動域。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.0, 0.0, 1.0, 0.8414709848078965)]
    [InlineData(0.5, 1.0, 0.1, 0.0028868935286866606)]
    [InlineData(1.0, -2.0, 2.0, 0.060126011786485786)]
    [InlineData(5.0, 2.0, 5.0, 0.09079533487543259)]
    [InlineData(0.0, 10.0, 1.0, 3.696585553278719e-11)]
    [InlineData(2.0, 10.0, 25.0, 1.2403044946508688)]
    [InlineData(1.0, 0.1, 100.0, -0.5335096827024138)]
    public void CoulombF_MatchesIndependentReferences(double l, double eta, double rho, double expected)
    {
        double actual = XMath.Special.CoulombF(l, eta, rho);
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
    [InlineData(double.NaN, 0.0, 1.0)]
    [InlineData(0.0, double.NaN, 1.0)]
    [InlineData(0.0, 0.0, double.NaN)]
    public void CoulombF_RejectsNaN(double l, double eta, double rho)
    {
        double actual = XMath.Special.CoulombF(l, eta, rho);
        Assert.True(double.IsNaN(actual));
    }

    // 自由波・非整数角運動量・障壁の内外・遠方の振動域。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.0, 0.0, 1.0, 0.5403023058681398)]
    [InlineData(0.5, 1.0, 0.1, 15.73023884056879)]
    [InlineData(1.0, -2.0, 2.0, -0.7942591281405753)]
    [InlineData(5.0, 2.0, 5.0, 5.629840456444406)]
    [InlineData(0.0, 10.0, 1.0, 3088184933.6735845)]
    [InlineData(2.0, 10.0, 25.0, -0.8375220311529138)]
    [InlineData(1.0, 0.1, 100.0, -0.8464450196970263)]
    public void CoulombG_MatchesIndependentReferences(double l, double eta, double rho, double expected)
    {
        double actual = XMath.Special.CoulombG(l, eta, rho);
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
    [InlineData(double.NaN, 0.0, 1.0)]
    [InlineData(0.0, double.NaN, 1.0)]
    [InlineData(0.0, 0.0, double.NaN)]
    public void CoulombG_RejectsNaN(double l, double eta, double rho)
    {
        double actual = XMath.Special.CoulombG(l, eta, rho);
        Assert.True(double.IsNaN(actual));
    }

    [Theory]
    [InlineData(.1)]
    [InlineData(1.0)]
    [InlineData(10.0)]
    public void FreeWavesAndDomains(double rho)
    {
        SpecialFunctionAssert.Close(Math.Sin(rho), XMath.Special.CoulombF(0, 0, rho), relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(Math.Cos(rho), XMath.Special.CoulombG(0, 0, rho), relativeTolerance: 2e-12);
        Assert.Equal(0, XMath.Special.CoulombF(0, 0, 0));
        Assert.Equal(1, XMath.Special.CoulombG(0, 0, 0));
        Assert.Equal(double.PositiveInfinity, XMath.Special.CoulombG(1, 0, 0));
        Assert.True(double.IsNaN(XMath.Special.CoulombF(-1, 0, rho)));
        Assert.True(double.IsNaN(XMath.Special.CoulombG(1, 0, -1)));
    }
}
