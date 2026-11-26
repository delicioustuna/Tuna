using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "HypergeometricU")]
public class HypergeometricUTests
{

    // 整数b付近・Temme/漸近展開の切替・終端多項式。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.5, 1.0, 0.1, 1.847102659887004)]
    [InlineData(1.0, 2.0, 2.0, 0.5)]
    [InlineData(2.0, 3.0, 10.0, 0.01)]
    [InlineData(-3.0, 2.0, 0.5, -8.875)]
    [InlineData(0.3, 0.7, 1.9999999999999998, 0.7602169263901107)]
    [InlineData(0.3, 0.7, 2.0000000000000004, 0.7602169263901107)]
    [InlineData(10.0, 20.0, 30.0, 2.86984877775011e-14)]
    [InlineData(1.0, 1.0000000000000002, 0.01, 4.078511443456428)]
    [InlineData(5.0, -2.0, 100.0, 6.87359317098509e-11)]
    public void HypergeometricU_MatchesIndependentReferences(double a, double b, double x, double expected)
    {
        double actual = XMath.Special.HypergeometricU(a, b, x);
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
    [InlineData(double.NaN, 1.0, 0.1)]
    [InlineData(0.5, double.NaN, 0.1)]
    [InlineData(0.5, 1.0, double.NaN)]
    public void HypergeometricU_RejectsNaN(double a, double b, double x)
    {
        double actual = XMath.Special.HypergeometricU(a, b, x);
        Assert.True(double.IsNaN(actual));
    }

    // Whittakerの独立参照値・次数の符号。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.0, 0.5, 0.1, 0.10004167187531003)]
    [InlineData(1.0, 0.3, 2.0, 0.40791958096953906)]
    [InlineData(-2.0, 1.0, 10.0, 10007.196198139334)]
    [InlineData(5.0, 2.0, 30.0, -11.041519078531376)]
    [InlineData(0.2, -0.3, 0.5, 0.6779854603982741)]
    public void WhittakerM_MatchesIndependentReferences(double kappa, double mu, double x, double expected)
    {
        double actual = XMath.Special.WhittakerM(kappa, mu, x);
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
    [InlineData(double.NaN, 0.5, 0.1)]
    [InlineData(0.0, double.NaN, 0.1)]
    [InlineData(0.0, 0.5, double.NaN)]
    public void WhittakerM_RejectsNaN(double kappa, double mu, double x)
    {
        double actual = XMath.Special.WhittakerM(kappa, mu, x);
        Assert.True(double.IsNaN(actual));
    }

    // Whittakerの独立参照値・次数の符号。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.0, 0.5, 0.1, 0.951229424500714)]
    [InlineData(1.0, 0.3, 2.0, 0.6787301515392412)]
    [InlineData(-2.0, 1.0, 10.0, 4.4367891687126396e-05)]
    [InlineData(5.0, 2.0, 30.0, 3.946574613995053)]
    [InlineData(0.2, -0.3, 0.5, 0.6779854603982741)]
    public void WhittakerW_MatchesIndependentReferences(double kappa, double mu, double x, double expected)
    {
        double actual = XMath.Special.WhittakerW(kappa, mu, x);
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
    [InlineData(double.NaN, 0.5, 0.1)]
    [InlineData(0.0, double.NaN, 0.1)]
    [InlineData(0.0, 0.5, double.NaN)]
    public void WhittakerW_RejectsNaN(double kappa, double mu, double x)
    {
        double actual = XMath.Special.WhittakerW(kappa, mu, x);
        Assert.True(double.IsNaN(actual));
    }

    [Theory]
    [InlineData(.1)]
    [InlineData(2.0)]
    [InlineData(100.0)]
    public void PowerClosedFormAndDomain(double x)
    {
        SpecialFunctionAssert.Close(Math.Pow(x, -.3), XMath.Special.HypergeometricU(.3, 1.3, x), relativeTolerance: 2e-12);
        Assert.Equal(1, XMath.Special.HypergeometricU(0, 2, x));
        Assert.True(double.IsNaN(XMath.Special.HypergeometricU(1, 2, -1)));
        Assert.True(double.IsNaN(XMath.Special.WhittakerM(1, .3, -1)));
        Assert.True(double.IsNaN(XMath.Special.WhittakerW(1, .3, -1)));
    }
}
