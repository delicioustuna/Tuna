using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "Polylogarithm")]
public class PolylogarithmTests
{

    // 整数/非整数次数・正負枝・退化した閉形式・極小の裾。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(2.0, 0.5, 0.5822405264650125)]
    [InlineData(0.5, -0.5, -0.37375223798097307)]
    [InlineData(-2.0, 0.3, 1.1370262390670554)]
    [InlineData(3.0, -10.0, -5.921064803756973)]
    [InlineData(0.3, -2.0, -0.8029335279883059)]
    [InlineData(5.0, 0.99, 1.026110477101306)]
    [InlineData(2.0, 1.0, 1.6449340668482264)]
    [InlineData(2.0, -1e+20, -1062.024556162528)]
    public void Polylogarithm_MatchesIndependentReferences(double s, double x, double expected)
    {
        double actual = XMath.Special.Polylogarithm(s, x);
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
    [InlineData(double.NaN, 0.5)]
    [InlineData(2.0, double.NaN)]
    public void Polylogarithm_RejectsNaN(double s, double x)
    {
        double actual = XMath.Special.Polylogarithm(s, x);
        Assert.True(double.IsNaN(actual));
    }

    // 整数/非整数次数・正負枝・退化した閉形式・極小の裾。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-1.0, -2.0, 0.11920292202211756)]
    [InlineData(0.0, 0.0, 0.6931471805599453)]
    [InlineData(0.5, -20.0, 2.0611536209365377e-09)]
    [InlineData(0.5, 2.0, 2.8237212774015843)]
    [InlineData(1.0, 10.0, 51.64488866743374)]
    [InlineData(2.0, 40.0, 10732.464029340596)]
    [InlineData(0.3, 100.0, 341.2433796420944)]
    [InlineData(0.5, -700.0, 9.85967654375977e-305)]
    public void FermiDirac_MatchesIndependentReferences(double s, double x, double expected)
    {
        double actual = XMath.Special.FermiDirac(s, x);
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
    [InlineData(double.NaN, -2.0)]
    [InlineData(-1.0, double.NaN)]
    public void FermiDirac_RejectsNaN(double s, double x)
    {
        double actual = XMath.Special.FermiDirac(s, x);
        Assert.True(double.IsNaN(actual));
    }

    // 整数/非整数次数・正負枝・退化した閉形式・極小の裾。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.0, -0.1, 2.3521684610440907)]
    [InlineData(0.5, -2.0, 0.14233433763743575)]
    [InlineData(2.0, -10.0, 4.540018741015351e-05)]
    [InlineData(2.0, 0.0, 1.2020569031595942)]
    [InlineData(-0.5, -0.1, 4.165296503346587)]
    [InlineData(1.0, -700.0, 9.85967654375977e-305)]
    public void BoseEinstein_MatchesIndependentReferences(double s, double x, double expected)
    {
        double actual = XMath.Special.BoseEinstein(s, x);
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
    [InlineData(double.NaN, -0.1)]
    [InlineData(0.0, double.NaN)]
    public void BoseEinstein_RejectsNaN(double s, double x)
    {
        double actual = XMath.Special.BoseEinstein(s, x);
        Assert.True(double.IsNaN(actual));
    }

    [Theory]
    [InlineData(-2.0)]
    [InlineData(.3)]
    [InlineData(.9)]
    public void ElementaryOrdersAndDomains(double x)
    {
        SpecialFunctionAssert.Close(x / (1 - x), XMath.Special.Polylogarithm(0, x), relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(-Math.Log(1 - x), XMath.Special.Polylogarithm(1, x), relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(Math.Log(1 + Math.Exp(x)), XMath.Special.FermiDirac(0, x), relativeTolerance: 2e-12);
        Assert.True(double.IsNaN(XMath.Special.Polylogarithm(.3, 1.1)));
        Assert.True(double.IsNaN(XMath.Special.BoseEinstein(1, .1)));
        Assert.Equal(0, XMath.Special.FermiDirac(1, double.NegativeInfinity));
        Assert.Equal(double.PositiveInfinity, XMath.Special.FermiDirac(1, double.PositiveInfinity));
        Assert.Equal(1, XMath.Special.FermiDirac(-1, double.PositiveInfinity));
        Assert.Equal(double.PositiveInfinity, XMath.Special.BoseEinstein(0, 0));
    }
}
