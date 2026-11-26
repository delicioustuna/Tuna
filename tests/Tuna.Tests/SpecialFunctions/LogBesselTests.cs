using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "LogBessel")]
public class LogBesselTests
{

    // 線形値の上下限外・大次数・比の安定性。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.0, 1e-10, 2.5000000000000002e-21, 0.0)]
    [InlineData(0.0, 1e-06, 2.4999999999998434e-13, 0.0)]
    [InlineData(0.0, 0.01, 2.499984375173609e-05, 0.0)]
    [InlineData(0.5, 1.0, -0.0643519910735318, 0.0)]
    [InlineData(10.0, 0.1, -45.06150803804031, 2e-12)]
    [InlineData(100.0, 20.0, -132.4955121081758, 2e-12)]
    [InlineData(1000.0, 1000.0, 528.2938870936566, 2e-12)]
    [InlineData(0.0, 10000.0, 9994.475903781433, 1.4551915228366852e-11)]
    public void LogBesselI_MatchesIndependentReferences(double nu, double x, double expected, double absoluteTolerance)
    {
        double actual = XMath.Special.LogBesselI(nu, x);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12, absoluteTolerance: absoluteTolerance);
            if (Math.Abs(expected) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expected), Math.Sign(actual));
            }
        }
    }

    // 各引数のNaNを独立に検査。
    [Theory]
    [InlineData(double.NaN, 1e-10)]
    [InlineData(0.0, double.NaN)]
    public void LogBesselI_RejectsNaN(double nu, double x)
    {
        double actual = XMath.Special.LogBesselI(nu, x);
        Assert.True(double.IsNaN(actual));
    }

    // 線形値の上下限外・大次数・比の安定性。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.0, 1e-10, -9.99999999975e-11, 0.0)]
    [InlineData(0.0, 1e-06, -9.9999975e-07, 0.0)]
    [InlineData(0.0, 0.01, -0.009975000156248264, 0.0)]
    [InlineData(0.5, 1.0, -1.0643519910735317, 2e-12)]
    [InlineData(10.0, 0.1, -45.1615080380403, 2e-12)]
    [InlineData(100.0, 20.0, -152.4955121081758, 2e-12)]
    [InlineData(1000.0, 1000.0, -471.7061129063434, 2e-12)]
    [InlineData(0.0, 10000.0, -5.524096218567699, 2e-12)]
    public void LogBesselIScaled_MatchesIndependentReferences(double nu, double x, double expected, double absoluteTolerance)
    {
        double actual = XMath.Special.LogBesselIScaled(nu, x);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12, absoluteTolerance: absoluteTolerance);
            if (Math.Abs(expected) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expected), Math.Sign(actual));
            }
        }
    }

    // 小さい |ν| の小引数で、1 + ν の丸めにより -γν が桁落ちしないことを確認する（参照値は mpmath 1.4.1）。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(false, 1e-20, 1e-10, -2.289178244559887e-19)]
    [InlineData(false, 1e-10, 1e-10, -2.3141782445656118e-09)]
    [InlineData(false, -1e-10, 0.001, 2.5070235307945757e-07)]
    [InlineData(false, 1e-06, 0.001, -6.7736878827318914e-06)]
    [InlineData(false, 1e-300, 1e-300, -6.908914594138721e-298)]
    [InlineData(true, 1e-20, 1e-300, -6.908914594138721e-18)]
    [InlineData(true, 1e-10, 1e-10, -2.414178244565612e-09)]
    public void LogBesselI_SmallOrderKeepsTheEulerTerm(bool scaled, double nu, double x, double expected)
    {
        double actual = scaled ? XMath.Special.LogBesselIScaled(nu, x) : XMath.Special.LogBesselI(nu, x);
        SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12);
    }

    // 各引数のNaNを独立に検査。
    [Theory]
    [InlineData(double.NaN, 1e-10)]
    [InlineData(0.0, double.NaN)]
    public void LogBesselIScaled_RejectsNaN(double nu, double x)
    {
        double actual = XMath.Special.LogBesselIScaled(nu, x);
        Assert.True(double.IsNaN(actual));
    }

    // 線形値の上下限外・大次数・比の安定性。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.0, 1e-10, 5e-11)]
    [InlineData(0.0, 1e-06, 4.999999999999375e-07)]
    [InlineData(0.0, 0.01, 0.004999937501041649)]
    [InlineData(0.5, 1.0, 0.3130352854993313)]
    [InlineData(10.0, 0.1, 0.004545368460310227)]
    [InlineData(100.0, 20.0, 0.09806687405574264)]
    [InlineData(1000.0, 1000.0, 0.413963628703407)]
    [InlineData(0.0, 10000.0, 0.999949998749875)]
    public void BesselIRatio_MatchesIndependentReferences(double nu, double x, double expected)
    {
        double actual = XMath.Special.BesselIRatio(nu, x);
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
    [InlineData(double.NaN, 1e-10)]
    [InlineData(0.0, double.NaN)]
    public void BesselIRatio_RejectsNaN(double nu, double x)
    {
        double actual = XMath.Special.BesselIRatio(nu, x);
        Assert.True(double.IsNaN(actual));
    }

    [Fact]
    public void OriginAndRatioRange()
    {
        Assert.Equal(0, XMath.Special.LogBesselI(0, 0));
        Assert.Equal(double.NegativeInfinity, XMath.Special.LogBesselI(1, 0));
        Assert.Equal(0, XMath.Special.LogBesselIScaled(0, 0));
        Assert.Equal(0, XMath.Special.BesselIRatio(0, 0));
        foreach (double x in new[] { .1, 1.0, 10.0, 1000.0 })
        {
            Assert.InRange(XMath.Special.BesselIRatio(2, x), 0, 1);
        }

        Assert.True(double.IsNaN(XMath.Special.LogBesselI(0, -1)));
        Assert.True(double.IsNaN(XMath.Special.LogBesselIScaled(0, -1)));
        Assert.True(double.IsNaN(XMath.Special.BesselIRatio(.5, -1)));
        Assert.Equal(-XMath.Special.BesselIRatio(0, 1), XMath.Special.BesselIRatio(0, -1));
    }

    // Debye の主要項 x/(ν+√(ν²+x²))（省略項 O(1/ν) は相対 1e-300 未満、参照値は 100/180 桁で一致を確認）。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAudit20261102")]
    [Theory]
    [InlineData(1e+307, 1e+307, 0.41421356237309503)]
    [InlineData(1e+308, 1e+307, 0.04987562112089027)]
    [InlineData(1e+308, 1e+308, 0.41421356237309503)]
    [InlineData(8e+307, 1.2e+308, 0.5351837584879964)]
    [InlineData(1e+308, 1.7e+308, 0.5719460543127071)]
    public void BesselIRatio_HugeOrderAndArgument_FixedReferences(double nu, double x, double expected)
    {
        double actual = XMath.Special.BesselIRatio(nu, x);
        if (double.IsInfinity(expected) || expected == 0)
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12,
                absoluteTolerance: Math.Abs(expected) < 2.2250738585072014e-308 ? 2 * double.Epsilon : 0);
        }
    }

    // Debye の主要項 √(ν²+x²) + ν log(x/(ν+√(ν²+x²)))（省略項 O(log ν) はこの入力では相対 1e-300 未満、参照値は 100/180 桁で一致を確認）。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAudit20261102")]
    [Theory]
    [InlineData(1e+307, 1e+307, 5.3283997535355204e+306)]
    [InlineData(1e+308, 1e+307, double.NegativeInfinity)]
    [InlineData(1e+308, 1e+308, 5.32839975353552e+307)]
    [InlineData(8e+307, 1.2e+308, 9.421044163852623e+307)]
    [InlineData(1e+308, 1.7e+308, 1.413597689639614e+308)]
    public void LogBesselI_HugeOrderAndArgument_FixedReferences(double nu, double x, double expected)
    {
        double actual = XMath.Special.LogBesselI(nu, x);
        if (double.IsInfinity(expected) || expected == 0)
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12,
                absoluteTolerance: Math.Abs(expected) < 2.2250738585072014e-308 ? 2 * double.Epsilon : 0);
        }
    }

    // 上の Debye 主要項から x を引いた参照。R+x の中間オーバーフローと、R 自体がオーバーフローする領域を検証する。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAudit20261102")]
    [Theory]
    [InlineData(1e+307, 1e+307, -4.6716002464644794e+306)]
    [InlineData(1e+308, 1e+307, double.NegativeInfinity)]
    [InlineData(1e+308, 1e+308, -4.67160024646448e+307)]
    [InlineData(8e+307, 1.2e+308, -2.578955836147376e+307)]
    [InlineData(1e+308, 1.7e+308, -2.8640231036038593e+307)]
    public void LogBesselIScaled_HugeOrderAndArgument_FixedReferences(double nu, double x, double expected)
    {
        double actual = XMath.Special.LogBesselIScaled(nu, x);
        if (double.IsInfinity(expected) || expected == 0)
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12,
                absoluteTolerance: Math.Abs(expected) < 2.2250738585072014e-308 ? 2 * double.Epsilon : 0);
        }
    }
}
