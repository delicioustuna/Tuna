using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "Synchrotron")]
public class SynchrotronTests
{

    // 正の定義積分・級数切替・指数的な裾。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.01, 0.44497250411421063)]
    [InlineData(0.25, 0.9157998878007424)]
    [InlineData(0.25000000000000006, 0.9157998878007424)]
    [InlineData(1.0, 0.651422815355364)]
    [InlineData(5.0, 0.021248129774981986)]
    [InlineData(50.0, 1.7347852039765936e-21)]
    public void SynchrotronF_MatchesIndependentReferences(double x, double expected)
    {
        double actual = XMath.Special.SynchrotronF(x);
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
    public void SynchrotronF_RejectsNaN(double x)
    {
        double actual = XMath.Special.SynchrotronF(x);
        Assert.True(double.IsNaN(actual));
    }

    // 級数切替の隣接値・非正規数の裾。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1e-20, 2.315509105382557e-07, 0.0)]
    [InlineData(0.1, 0.47529626776208206, 0.0)]
    [InlineData(0.49999999999999994, 0.6029652323601679, 0.0)]
    [InlineData(0.5, 0.6029652323601679, 0.0)]
    [InlineData(0.5000000000000001, 0.6029652323601679, 0.0)]
    [InlineData(1.0, 0.4944750621042083, 0.0)]
    [InlineData(50.0, 1.712604265071687e-21, 0.0)]
    [InlineData(700.0, 3.269880654608996e-303, 0.0)]
    [InlineData(740.0, 1.4283e-320, 1e-323)]
    public void SynchrotronG_MatchesIndependentReferences(double x, double expected, double absoluteTolerance)
    {
        double actual = XMath.Special.SynchrotronG(x);
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
    [InlineData(double.NaN)]
    public void SynchrotronG_RejectsNaN(double x)
    {
        double actual = XMath.Special.SynchrotronG(x);
        Assert.True(double.IsNaN(actual));
    }

    [Fact]
    public void OriginAndDomains()
    {
        Assert.Equal(0, XMath.Special.SynchrotronF(0));
        Assert.Equal(0, XMath.Special.SynchrotronG(0));
        Assert.Equal(0, XMath.Special.SynchrotronF(double.PositiveInfinity));
        Assert.Equal(0, XMath.Special.SynchrotronG(double.PositiveInfinity));
        Assert.True(double.IsNaN(XMath.Special.SynchrotronF(-1)));
        Assert.True(double.IsNaN(XMath.Special.SynchrotronG(-1)));
        foreach (double x in new[] { .01, .1, 1.0, 10.0 })
        {
            Assert.True(XMath.Special.SynchrotronF(x) > 0);
            Assert.True(XMath.Special.SynchrotronG(x) > 0);
        }
    }
}
