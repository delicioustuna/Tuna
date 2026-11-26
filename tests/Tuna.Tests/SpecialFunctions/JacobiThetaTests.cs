using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "JacobiTheta")]
public class JacobiThetaTests
{

    // 直接級数/虚数変換の切替・高nome・偶奇と周期。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.0, 0.1, 0.0)]
    [InlineData(0.3, 0.001, 0.10510322115643855)]
    [InlineData(1.0, 0.5, 1.3303784981792746)]
    [InlineData(-1.0, 0.5, -1.3303784981792746)]
    [InlineData(0.25, 0.99, 7.313634745491505e-75)]
    [InlineData(2.0, 0.9, 0.9503886199654177)]
    [InlineData(0.2, 0.043213918263772244, 0.18020033394788462)]
    [InlineData(0.2, 0.04321391826377225, 0.18020033394788462)]
    [InlineData(0.2, 0.04321391826377226, 0.18020033394788462)]
    public void JacobiTheta1_MatchesIndependentReferences(double z, double q, double expected)
    {
        double actual = XMath.Special.JacobiTheta1(z, q);
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
    [InlineData(0.0, double.NaN)]
    public void JacobiTheta1_RejectsNaN(double z, double q)
    {
        double actual = XMath.Special.JacobiTheta1(z, q);
        Assert.True(double.IsNaN(actual));
    }

    // 直接級数/虚数変換の切替・高nome・偶奇と周期。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.0, 0.1, 1.1359306015682802)]
    [InlineData(0.3, 0.001, 0.3397712627334292)]
    [InlineData(1.0, 0.5, 0.5001981385144562)]
    [InlineData(-1.0, 0.5, 0.5001981385144562)]
    [InlineData(0.25, 0.99, 0.035215882235151014)]
    [InlineData(2.0, 0.9, -2.3191330483928617e-05)]
    [InlineData(0.2, 0.043213918263772244, 0.8951048891589518)]
    [InlineData(0.2, 0.04321391826377225, 0.8951048891589518)]
    [InlineData(0.2, 0.04321391826377226, 0.8951048891589518)]
    public void JacobiTheta2_MatchesIndependentReferences(double z, double q, double expected)
    {
        double actual = XMath.Special.JacobiTheta2(z, q);
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
    [InlineData(0.0, double.NaN)]
    public void JacobiTheta2_RejectsNaN(double z, double q)
    {
        double actual = XMath.Special.JacobiTheta2(z, q);
        Assert.True(double.IsNaN(actual));
    }

    // 直接級数/虚数変換の切替・高nome・偶奇と周期。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.0, 0.1, 1.2002000020000003)]
    [InlineData(0.3, 0.001, 1.001650671230544)]
    [InlineData(1.0, 0.5, 0.5058938857304847)]
    [InlineData(-1.0, 0.5, 0.5058938857304847)]
    [InlineData(0.25, 0.99, 0.035215882235151014)]
    [InlineData(2.0, 0.9, 2.3191330484283696e-05)]
    [InlineData(0.2, 0.043213918263772244, 1.0796101683315946)]
    [InlineData(0.2, 0.04321391826377225, 1.0796101683315946)]
    [InlineData(0.2, 0.04321391826377226, 1.0796101683315946)]
    public void JacobiTheta3_MatchesIndependentReferences(double z, double q, double expected)
    {
        double actual = XMath.Special.JacobiTheta3(z, q);
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
    [InlineData(0.0, double.NaN)]
    public void JacobiTheta3_RejectsNaN(double z, double q)
    {
        double actual = XMath.Special.JacobiTheta3(z, q);
        Assert.True(double.IsNaN(actual));
    }

    // 直接級数/虚数変換の切替・高nome・偶奇と周期。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.0, 0.1, 0.8001999980000002)]
    [InlineData(0.3, 0.001, 0.9983493287709053)]
    [InlineData(1.0, 0.5, 1.3306863284854333)]
    [InlineData(-1.0, 0.5, 1.3306863284854333)]
    [InlineData(0.25, 0.99, 7.313634745491505e-75)]
    [InlineData(2.0, 0.9, 0.9503886199654177)]
    [InlineData(0.2, 0.043213918263772244, 0.9203995502876748)]
    [InlineData(0.2, 0.04321391826377225, 0.9203995502876748)]
    [InlineData(0.2, 0.04321391826377226, 0.9203995502876748)]
    public void JacobiTheta4_MatchesIndependentReferences(double z, double q, double expected)
    {
        double actual = XMath.Special.JacobiTheta4(z, q);
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
    [InlineData(0.0, double.NaN)]
    public void JacobiTheta4_RejectsNaN(double z, double q)
    {
        double actual = XMath.Special.JacobiTheta4(z, q);
        Assert.True(double.IsNaN(actual));
    }

    [Theory]
    [InlineData(.01)]
    [InlineData(.5)]
    [InlineData(.9)]
    public void ParityAndJacobiIdentity(double q)
    {
        Assert.Equal(-XMath.Special.JacobiTheta1(.3, q), XMath.Special.JacobiTheta1(-.3, q));
        SpecialFunctionAssert.Close(XMath.Special.JacobiTheta2(.3, q), XMath.Special.JacobiTheta2(-.3, q), relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(XMath.Special.JacobiTheta3(.3, q), XMath.Special.JacobiTheta3(-.3, q), relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(XMath.Special.JacobiTheta4(.3, q), XMath.Special.JacobiTheta4(-.3, q), relativeTolerance: 2e-12);
        double t2 = XMath.Special.JacobiTheta2(0, q), t3 = XMath.Special.JacobiTheta3(0, q), t4 = XMath.Special.JacobiTheta4(0, q);
        SpecialFunctionAssert.Close(Math.Pow(t3, 4), Math.Pow(t2, 4) + Math.Pow(t4, 4), relativeTolerance: 2e-12);
        Assert.Equal(0, XMath.Special.JacobiTheta1(.3, 0));
        Assert.Equal(0, XMath.Special.JacobiTheta2(.3, 0));
        Assert.Equal(1, XMath.Special.JacobiTheta3(.3, 0));
        Assert.Equal(1, XMath.Special.JacobiTheta4(.3, 0));
        Assert.True(double.IsNaN(XMath.Special.JacobiTheta1(.3, -.1)));
        Assert.True(double.IsNaN(XMath.Special.JacobiTheta2(.3, 1)));
        Assert.True(double.IsNaN(XMath.Special.JacobiTheta3(.3, 1.1)));
        Assert.True(double.IsNaN(XMath.Special.JacobiTheta4(.3, 1.1)));
    }

    // 正負の大きな引数での π による周期還元を固定参照値と検証する。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditFollowup20261102")]
    [Theory]
    [InlineData(-1e+16, 0.5, -1.099455588022218)]
    [InlineData(1e+16, 0.5, 1.099455588022218)]
    [InlineData(-1e+18, 0.5, 2.0861269068614714)]
    [InlineData(1e+18, 0.5, -2.0861269068614714)]
    [InlineData(-1e+20, 0.5, 0.7141268176113005)]
    [InlineData(1e+20, 0.5, -0.7141268176113005)]
    [InlineData(1e+20, 0.01, -0.4080382807330806)]
    public void JacobiTheta1_PeriodReduction_FixedReferences(double z, double q, double expected)
    {
        double actual = XMath.Special.JacobiTheta1(z, q);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12,
                absoluteTolerance: Math.Abs(expected) < 2.2250738585072014e-308 ? 2 * double.Epsilon : 0);
        }
    }

    // 正負の大きな引数での π による周期還元を固定参照値と検証する。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditFollowup20261102")]
    [Theory]
    [InlineData(-1e+16, 0.5, -0.6702961068429049)]
    [InlineData(1e+16, 0.5, -0.6702961068429049)]
    [InlineData(-1e+18, 0.5, 0.06694797713871103)]
    [InlineData(1e+18, 0.5, 0.06694797713871103)]
    [InlineData(-1e+20, 0.5, 1.046644484153818)]
    [InlineData(1e+20, 0.5, 1.046644484153818)]
    [InlineData(1e+20, 0.01, 0.4831451581310024)]
    public void JacobiTheta2_PeriodReduction_FixedReferences(double z, double q, double expected)
    {
        double actual = XMath.Special.JacobiTheta2(z, q);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12,
                absoluteTolerance: Math.Abs(expected) < 2.2250738585072014e-308 ? 2 * double.Epsilon : 0);
        }
    }

    // 正負の大きな引数での π による周期還元を固定参照値と検証する。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditFollowup20261102")]
    [Theory]
    [InlineData(-1e+16, 0.5, 0.6732102806644252)]
    [InlineData(1e+16, 0.5, 0.6732102806644252)]
    [InlineData(-1e+18, 0.5, 0.13626850615102298)]
    [InlineData(1e+18, 0.5, 0.13626850615102298)]
    [InlineData(-1e+20, 0.5, 1.047435532979688)]
    [InlineData(1e+20, 0.5, 1.047435532979688)]
    [InlineData(1e+20, 0.01, 1.0033460122741067)]
    public void JacobiTheta3_PeriodReduction_FixedReferences(double z, double q, double expected)
    {
        double actual = XMath.Special.JacobiTheta3(z, q);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12,
                absoluteTolerance: Math.Abs(expected) < 2.2250738585072014e-308 ? 2 * double.Epsilon : 0);
        }
    }

    // 正負の大きな引数での π による周期還元を固定参照値と検証する。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditFollowup20261102")]
    [Theory]
    [InlineData(-1e+16, 0.5, 1.1001196676948017)]
    [InlineData(1e+16, 0.5, 1.1001196676948017)]
    [InlineData(-1e+18, 0.5, 2.0861358487283885)]
    [InlineData(1e+18, 0.5, 2.0861358487283885)]
    [InlineData(-1e+20, 0.5, 0.7166071233882004)]
    [InlineData(1e+20, 0.5, 0.7166071233882004)]
    [InlineData(1e+20, 0.01, 0.9966539499650782)]
    public void JacobiTheta4_PeriodReduction_FixedReferences(double z, double q, double expected)
    {
        double actual = XMath.Special.JacobiTheta4(z, q);
        if (expected == 0 || double.IsInfinity(expected))
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
