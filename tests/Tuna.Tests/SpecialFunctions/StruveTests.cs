using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "Struve")]
public class StruveTests
{

    // 負次数・半整数の閉形式・負引数の整数枝・級数/積分/漸近域。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-2.0, 0.5, -1.374100482349553)]
    [InlineData(-1.5, 2.0, -0.49129377868716234)]
    [InlineData(-0.5, 1.0, 0.6713967071418031)]
    [InlineData(0.0, 0.01, 0.006366126988428492)]
    [InlineData(1.0, -2.0, 0.6467637282835621)]
    [InlineData(0.3, 5.0, 0.05475134668694588)]
    [InlineData(5.0, 20.0, 110.24836574556309)]
    [InlineData(0.0, 50.0, -0.085337674826119)]
    [InlineData(20.0, 100.0, 199833871304281.5)]
    public void StruveH_MatchesIndependentReferences(double nu, double x, double expected)
    {
        double actual = XMath.Special.StruveH(nu, x);
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
    [InlineData(-2.0, double.NaN)]
    public void StruveH_RejectsNaN(double nu, double x)
    {
        double actual = XMath.Special.StruveH(nu, x);
        Assert.True(double.IsNaN(actual));
    }

    // 負次数・半整数の閉形式・負引数の整数枝・級数/積分/漸近域。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-2.0, 0.5, -1.1617675758350725)]
    [InlineData(-1.5, 2.0, 1.0994731886331097)]
    [InlineData(-0.5, 1.0, 0.9376748882454876)]
    [InlineData(0.0, 0.01, 0.006366268459489019)]
    [InlineData(1.0, -2.0, 1.1027597873677157)]
    [InlineData(0.3, 5.0, 26.703611002950076)]
    [InlineData(5.0, 20.0, 23018286.731382865)]
    [InlineData(0.0, 50.0, 2.9325537838493362e+20)]
    [InlineData(20.0, 100.0, 1.4483461256427172e+41)]
    public void StruveL_MatchesIndependentReferences(double nu, double x, double expected)
    {
        double actual = XMath.Special.StruveL(nu, x);
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
    [InlineData(-2.0, double.NaN)]
    public void StruveL_RejectsNaN(double nu, double x)
    {
        double actual = XMath.Special.StruveL(nu, x);
        Assert.True(double.IsNaN(actual));
    }

    [Theory]
    [InlineData(0, .2)]
    [InlineData(1, 2.0)]
    [InlineData(3, 10.0)]
    public void IntegerArgumentParity(int n, double x)
    {
        double sign = n % 2 == 0 ? -1 : 1;
        Assert.Equal(sign * XMath.Special.StruveH(n, x), XMath.Special.StruveH(n, -x));
        Assert.Equal(sign * XMath.Special.StruveL(n, x), XMath.Special.StruveL(n, -x));
        Assert.Equal(0, XMath.Special.StruveH(n, 0));
        Assert.Equal(0, XMath.Special.StruveL(n, 0));
        Assert.True(double.IsNaN(XMath.Special.StruveH(.3, -1)));
        Assert.True(double.IsNaN(XMath.Special.StruveL(.3, -1)));
    }
}
