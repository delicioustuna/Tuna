using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "ParabolicCylinder")]
public class ParabolicCylinderTests
{

    // Hermiteの閉形式・原点・負引数接続公式・転回点。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-0.5, 0.1, 0.9975031223974601)]
    [InlineData(0.3, 0.0, 1.257002390009113)]
    [InlineData(0.3, -2.0, 5.155868974984675)]
    [InlineData(2.0, 10.0, 4.2106240702288184e-14)]
    [InlineData(-3.0, 2.0, 1.0920159712050257)]
    [InlineData(-20.0, 9.0, 300341051.21369654)]
    [InlineData(20.0, -5.0, 10.576836532636584)]
    [InlineData(100.0, 20.0, 1.962162941335358e-179)]
    public void ParabolicCylinderU_MatchesIndependentReferences(double a, double x, double expected)
    {
        double actual = XMath.Special.ParabolicCylinderU(a, x);
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
    [InlineData(double.NaN, 0.1)]
    [InlineData(-0.5, double.NaN)]
    public void ParabolicCylinderU_RejectsNaN(double a, double x)
    {
        double actual = XMath.Special.ParabolicCylinderU(a, x);
        Assert.True(double.IsNaN(actual));
    }

    // Hermiteの閉形式・原点・負引数接続公式・転回点。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-0.5, 0.1, 0.07972208200499908)]
    [InlineData(0.3, 0.0, 0.8426894498302632)]
    [InlineData(0.3, -2.0, 1.6148122086847971)]
    [InlineData(2.0, 10.0, 1823604920585.2852)]
    [InlineData(-3.0, 2.0, -0.24003140886983018)]
    [InlineData(-20.0, 9.0, 1.0998915864160147e-09)]
    [InlineData(20.0, -5.0, 0.007361624201647628)]
    [InlineData(100.0, 20.0, 1.4376731718696684e+177)]
    public void ParabolicCylinderV_MatchesIndependentReferences(double a, double x, double expected)
    {
        double actual = XMath.Special.ParabolicCylinderV(a, x);
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
    [InlineData(double.NaN, 0.1)]
    [InlineData(-0.5, double.NaN)]
    public void ParabolicCylinderV_RejectsNaN(double a, double x)
    {
        double actual = XMath.Special.ParabolicCylinderV(a, x);
        Assert.True(double.IsNaN(actual));
    }

    // Hermiteの閉形式・原点・負引数接続公式・転回点。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-0.5, 0.1, 1.1581458438027878)]
    [InlineData(0.3, 0.0, 0.772406581872622)]
    [InlineData(0.3, -2.0, -0.5811514680907444)]
    [InlineData(2.0, 10.0, 1.374906442631438e-09)]
    [InlineData(-3.0, 2.0, 0.019653250321884745)]
    [InlineData(-20.0, 9.0, 1.6080459540795413e-29)]
    [InlineData(20.0, -5.0, -608550556.6427445)]
    [InlineData(100.0, 20.0, 3.988455852340775e+78)]
    public void ParabolicCylinderD_MatchesIndependentReferences(double a, double x, double expected)
    {
        double actual = XMath.Special.ParabolicCylinderD(a, x);
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
    [InlineData(double.NaN, 0.1)]
    [InlineData(-0.5, double.NaN)]
    public void ParabolicCylinderD_RejectsNaN(double a, double x)
    {
        double actual = XMath.Special.ParabolicCylinderD(a, x);
        Assert.True(double.IsNaN(actual));
    }

    [Theory]
    [InlineData(-2.0)]
    [InlineData(0.0)]
    [InlineData(2.0)]
    public void HermiteClosedForms(double x)
    {
        SpecialFunctionAssert.Close(Math.Exp(-x * x / 4), XMath.Special.ParabolicCylinderD(0, x), relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(x * Math.Exp(-x * x / 4), XMath.Special.ParabolicCylinderD(1, x), relativeTolerance: 2e-12);
        Assert.Equal(0, XMath.Special.ParabolicCylinderU(1, double.PositiveInfinity));
        Assert.Equal(double.PositiveInfinity, XMath.Special.ParabolicCylinderV(1, double.PositiveInfinity));
    }
}
